import math,glfw,os,numpy as np
from .shader_program import ShaderProgram,report_GL_error
from .shader_mesh import ShaderMesh
from OpenGL import GL
from PIL import Image

class Visibility:
    '''
        This class computes visibility of a triangle mesh using GPU rasterization
        Input: a ShaderMesh object
        Input: the resolution of rasterization
        Input: whether to compute triangle id on GPU or CPU
        Input: create a window or use a headless application
    '''
    def __init__(self, mesh, res=512, useShader=True, createWnd=True):
        if isinstance(mesh,str):
            self.mesh=ShaderMesh(mesh)
        else: self.mesh=mesh
        if createWnd:
            self.wnd=None
        self.res=res
        self.useShader=useShader
        self.init_visibility()
        
    def __del__(self):
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
        if hasattr(self,"FBORENDER"):
            GL.glDeleteTextures(self.TARGET)
            GL.glDeleteRenderbuffers(6,self.DEPTH)
            GL.glDeleteFramebuffers(1,[self.FBORENDER])
            report_GL_error()
        
        if hasattr(self,"FBOACCUM"):
            GL.glDeleteTextures([self.ACCUM])
            GL.glDeleteTextures([self.AREA])
            GL.glDeleteFramebuffers(1,[self.FBOACCUM])
            del self.prog
            report_GL_error()
        if hasattr(self,"SANGLE"):
            GL.glDeleteTextures([self.SANGLE])
            report_GL_error()
        
        if self.mesh is not None:
            del self.mesh
            report_GL_error()
        
        if hasattr(self,"wnd"):
            glfw.destroy_window(self.wnd)
            glfw.terminate()
        
    '''
        Create internal off-screen render buffer to store temporary data
    '''
    def init_visibility(self):
        if hasattr(self,"wnd"):
            glfw.init()
            glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
            self.wnd = glfw.create_window(self.res, self.res, "", None, None)
            glfw.make_context_current(self.wnd)
        
        self.resMax = GL.glGetIntegerv(GL.GL_MAX_TEXTURE_SIZE)
        if self.res>self.resMax:
            raise RuntimeError("res(%d) must be less than the hardware limit (%d)!"%(self.res,self.resMax))
        
        if self.mesh is not None:
            self.mesh.init_visibility(self.useShader)
        self.create_render_FBO()
        if self.mesh is not None:
            self.create_accumulate_FBO()
            self.create_accumulate_shader()
        self.create_solid_angle_texture()
        
    '''
        Used internally by init_visibility, create 6 off-screen render buffers to form a cubemap (with bound depth map)
    '''
    def create_render_FBO(self):
        #create a color buffer
        self.TARGET = [GL.glGenTextures(1) for i in range(6)]
        for d in range(6):
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.TARGET[d])
            GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGB32F, self.res, self.res, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, None)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST);
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST); 
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE);
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE);
            report_GL_error()
            
        #create a render buffer
        self.DEPTH = [GL.glGenRenderbuffers(1) for i in range(6)]
        for d in range(6):
            GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, self.DEPTH[d])
            GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_DEPTH_COMPONENT32, self.res, self.res)
            report_GL_error()
        
        #create a frame buffer
        self.FBORENDER = GL.glGenFramebuffers(1)
        report_GL_error()
        
    '''
        Used internally by init_visibility, create a linear buffer to store visibility of each triangle
        After rendering the data of this buffer is read back to CPU
    '''
    def create_accumulate_FBO(self):
        nrTriangle = len(self.mesh.mesh.faces)
        self.szBuffer = math.ceil(math.sqrt(nrTriangle))
        if self.szBuffer>self.resMax:
            raise RuntimeError("#triangle(%d) must be less than the hardware limit (%d)!"%(nrTriangle,self.resMax*self.resMax))
        
        self.ACCUM = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.ACCUM)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RG32F, self.szBuffer, self.szBuffer, 0, GL.GL_RGB, GL.GL_UNSIGNED_BYTE, None)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST);
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST); 
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE);
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE);
        report_GL_error()
        
        areas=self.mesh.area().tolist()
        while len(areas)<self.szBuffer*self.szBuffer:
            areas.append(0.)
        areas=np.array(areas).astype(GL.GLfloat)
        self.AREA = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.AREA)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RG32F, self.szBuffer, self.szBuffer, 0, GL.GL_RED, GL.GL_FLOAT, areas)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST);
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST); 
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE);
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE);
        report_GL_error()
        
        #create a frame buffer
        self.FBOACCUM = GL.glGenFramebuffers(1)
        report_GL_error()
        
    '''
        Used internally by init_visibility, create a grayscale texture that stores the solid angle each pixel occupies
    '''
    def create_solid_angle_texture(self):
        self.SANGLE = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.SANGLE)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGB32F, self.res, self.res, 0, GL.GL_RED, GL.GL_FLOAT, None)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST);
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST); 
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE);
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE);
        report_GL_error()
        
        #setup render target
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.FBORENDER)
        GL.glFramebufferTexture(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, self.SANGLE, 0)
        GL.glViewport(0,0,self.res,self.res)
        GL.glScissor(0,0,self.res,self.res)
        report_GL_error()
        
        #render mesh
        GL.glDisable(GL.GL_BLEND)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDepthFunc(GL.GL_LESS)
        GL.glClearColor(0, 0, 0, 0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)
        report_GL_error()
        
        progSolidAngle=self.create_solid_angle_shader()
        with progSolidAngle:
            GL.glUniform1i(GL.glGetUniformLocation(progSolidAngle.prog,"res"),self.res)
            GL.glEnableClientState(GL.GL_VERTEX_ARRAY)
            GL.glVertexPointer(3,GL.GL_FLOAT,0,[0.,0.,0.])
            GL.glDrawArraysInstanced(GL.GL_POINTS,0,1,self.res*self.res)
            GL.glDisableClientState(GL.GL_VERTEX_ARRAY)
        report_GL_error()
        del progSolidAngle
        
        GL.glFramebufferTexture(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, self.TARGET[0], 0)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
        if not os.path.exists("output"):
            os.mkdir("output")
        Visibility.draw_solid_angle_texture("output/solidAngle.png",self.SANGLE)
        
    '''
        Used internally by create_solid_angle_texture, fill data into the texture buffer
        Input: texture id
    '''
    @staticmethod
    def draw_solid_angle_texture(path, id):
        GL.glBindTexture(GL.GL_TEXTURE_2D, id)
        str = GL.glGetTexImage(GL.GL_TEXTURE_2D, 0, GL.GL_RGB, GL.GL_FLOAT)
        w = GL.glGetTexLevelParameteriv(GL.GL_TEXTURE_2D, 0, GL.GL_TEXTURE_WIDTH)
        h = GL.glGetTexLevelParameteriv(GL.GL_TEXTURE_2D, 0, GL.GL_TEXTURE_HEIGHT)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        
        data = np.fromstring(str, dtype=np.float32)
        maxAngle = np.amax(data)
        data /= maxAngle
        data = (data*255).astype(np.uint8)
        
        if path is not None:
            data = data.reshape((w,h,3))
            img = Image.fromarray(data)
            img.save(path)
        return maxAngle
           
    '''
        Used internally by create_solid_angle_texture, create a shader that computes the pixel-wise solid angle
    ''' 
    def create_solid_angle_shader(self):
        vert='''
            #version 410 compatibility
            uniform int res;
            in int gl_InstanceID;
            out vec4 gl_Position;
            flat out int x;
            flat out int y;
            void main()
            {
                x=gl_InstanceID/res;
                y=gl_InstanceID%res;
                gl_Position=vec4(float(2*x+1)/float(res)-1,float(2*y+1)/float(res)-1,0,1); 
            }
        '''
        frag='''
            #version 410 compatibility
            uniform int res;
            flat in int x;
            flat in int y;
            in float solidAngle;
            out vec4 FragColor;
            float solid_angle_triangle(vec3 u1,vec3 u2,vec3 u3)
            {
                u1=normalize(u1);
                u2=normalize(u2);
                u3=normalize(u3);
                return 2*atan(dot(cross(u1,u2),u3)/(1+dot(u2,u3)+dot(u1,u2)+dot(u1,u3)));
            }
            float solid_angle_pixel(int x,int y)
            {
                float alphax1=x/float(res),alphax2=(x+1)/float(res);
                float alphay1=y/float(res),alphay2=(y+1)/float(res);
                vec3 u0=vec3(-1.*(1-alphax1)+1.*alphax1,-1.*(1-alphay1)+1.*alphay1,1.);
                vec3 u1=vec3(-1.*(1-alphax2)+1.*alphax2,-1.*(1-alphay1)+1.*alphay1,1.);
                vec3 u2=vec3(-1.*(1-alphax2)+1.*alphax2,-1.*(1-alphay2)+1.*alphay2,1.);
                vec3 u3=vec3(-1.*(1-alphax1)+1.*alphax1,-1.*(1-alphay2)+1.*alphay2,1.);
                return solid_angle_triangle(u0,u1,u2)+solid_angle_triangle(u0,u2,u3);
            }
            void main()
            {
                float s=solid_angle_pixel(x,y);
                FragColor=vec4(s,s,s,1);
            } 
        '''
        return ShaderProgram(vert=vert,frag=frag)
                       
    '''
        Used internally by init_visibility, create a shader that perform the accumulate from the cubemap to the linear buffer
    ''' 
    def create_accumulate_shader(self):
        vert='''
            #version 410 compatibility
            uniform int res;
            in int gl_InstanceID;
            out vec4 gl_Position;
            void main()
            {
                gl_Position=vec4(float(gl_InstanceID/res),float(gl_InstanceID%res),0,1); 
            }
        '''
        geom='''
            #version 410 compatibility
            uniform sampler2D area_input;
            uniform sampler2D solid_angle_input;
            uniform sampler2D color_input[6];
            uniform int szBuffer;
            layout(points) in;
            layout(points,max_vertices=6) out;
            out float solidAngle;
            out float solidAngleByArea;
            int fetchId(vec4 pixel) 
            {
                return int(pixel.r*255)*(255*255)+int(pixel.g*255)*255+int(pixel.b*255);
            }
            void main()
            {
                ivec2 coord=ivec2(int(gl_in[0].gl_Position.x),int(gl_in[0].gl_Position.y));
                float tmpSolidAngle=texelFetch(solid_angle_input,coord,0).x;
                for(int i=0;i<6;i++) {
                    vec4 pixel=texelFetch(color_input[i],coord,0);
                    int id=fetchId(pixel)-1;
                    if(id>=0) {
                        int x=id%szBuffer;
                        int y=id/szBuffer;
                        
                        solidAngle=tmpSolidAngle;
                        solidAngleByArea=tmpSolidAngle/texelFetch(area_input,ivec2(x,y),0).x;
                        gl_Position=vec4(float(2*x+1)/float(szBuffer)-1,float(2*y+1)/float(szBuffer)-1,0,1);
                        EmitVertex();
                        EndPrimitive();
                    }
                }
            }
        '''
        frag='''
            #version 410 compatibility
            in float solidAngle;
            in float solidAngleByArea;
            out vec4 FragColor;
            void main()
            {
                FragColor=vec4(solidAngle,solidAngleByArea,1,1);
            } 
        '''
        self.prog=ShaderProgram(vert=vert,geom=geom,frag=frag)
        
    '''
        Add a robot as an occluder for the light
        Input: a klampt.RobotModel object
        Input: a link id on the robot that represents the bulb
    '''
    def set_robot(self,robot,bulbId):
        if robot is None:
            if hasattr(self,'robotOccluder'):
                delattr(self,'robotOccluder')
        else: 
            from . import robot_occluder
            self.robotOccluder=robot_occluder.Occluder(robot,bulbId)
      
    '''      
        main API for calculating visibility
        Input: choose to only render triangle indices in range id0<=id<id1
        Input: camera position
        Input: whether clear the accumulation buffer before rendering
        Input: whether perform accumulation on GPU
        
        if your environment has N meshes, you can render the entire mesh by setting id0=id1=None
        if you want to render part of the mesh, set 0<=id0<id1<#triangles
        this function works in three modes:
        1. single point without robot:
            in this case, pos is a list of 3 floats representing the light position
        2. multiple points without robot:
            in this case, pos is a list of list of 3 floats representing multiple light positions
            the calculated radiance will be accumulated
        3. robot occluder:
            if you have a robot as occluder, then call: visibility.set_robot(robot)
            here the parameter of the robot can be a path to .rob file or a klampt.RobotModel object
            after calling set_robot, pos must be a configuration of the robot
            you can also set pos=None and the current configuration will be used
    '''
    def render(self, id0, id1, pos, cleared=False, accumCPU=False):
        if hasattr(self,'robotOccluder'):
            if pos is not None:
                self.robotOccluder.set_config(pos)
            poses=self.robotOccluder.get_light_positions()
        else: poses=pos if isinstance(pos[0],list) else [pos]
        for pos in poses:
            pos=np.array(pos)
            zNear, zFar = self.mesh.compute_zNear_zFar(pos)
            for d in range(6):
                #setup render target
                GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.FBORENDER)
                GL.glFramebufferTexture(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, self.TARGET[d], 0)
                GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, GL.GL_DEPTH_ATTACHMENT, GL.GL_RENDERBUFFER, self.DEPTH[d])
                GL.glViewport(0,0,self.res,self.res)
                GL.glScissor(0,0,self.res,self.res)
                report_GL_error()
                
                Visibility.set_view_projection_matrix_cube(zNear, zFar, d, pos)
            
                #render mesh
                GL.glDisable(GL.GL_BLEND)
                GL.glEnable(GL.GL_DEPTH_TEST)
                GL.glDepthFunc(GL.GL_LESS)
                GL.glClearColor(0, 0, 0, 0)
                GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
                self.mesh.draw_visibility(id0, id1, pos)
                if hasattr(self,'robotOccluder'):
                    self.robotOccluder.drawGL()
                report_GL_error()
            cleared=self.accumulate_GPU(cleared)
             
        if accumCPU:
            assert len(poses)==1
            return self.accumulate_CPU()
        else: return self.read_accumulatebuffer()
              
    '''
        Used internally by render, perform accumulation on CPU
    ''' 
    def accumulate_CPU(self):
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.SANGLE)
        str = GL.glGetTexImage(GL.GL_TEXTURE_2D, 0, GL.GL_RGB, GL.GL_FLOAT)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        solidAngleData = np.fromstring(str, dtype=np.float32).reshape((self.res,self.res,3))
        
        solidAngle=[0. for i in self.mesh.mesh.faces]
        for id in range(6):
            data=self.read_renderbuffer(id).astype(np.uint32)
            data=data[:,:,0]*255*255+data[:,:,1]*255+data[:,:,2]
            for r in range(data.shape[0]):
                for c in range(data.shape[1]):
                    if data[r,c]>0:
                        solidAngle[data[r,c]-1]+=solidAngleData[r,c,0]
        return np.array(solidAngle),np.array(solidAngle)/self.mesh.area()
           
    '''
        Used internally by render, perform accumulation on GPU
    ''' 
    def accumulate_GPU(self, cleared):
        #setup accumulate target
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.FBOACCUM)
        GL.glFramebufferTexture(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, self.ACCUM, 0)
        GL.glViewport(0,0,self.szBuffer,self.szBuffer)
        GL.glScissor(0,0,self.szBuffer,self.szBuffer)
             
        #render accumulate
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_ONE, GL.GL_ONE)
        GL.glDisable(GL.GL_DEPTH_TEST)
        if not cleared:
            GL.glClearColor(0, 0, 0, 0)
            cleared=True
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)
        with self.prog:
            #input area
            GL.glActiveTexture(GL.GL_TEXTURE7)
            GL.glBindTexture(GL.GL_TEXTURE_2D,self.AREA)
            GL.glUniform1i(GL.glGetUniformLocation(self.prog.prog,"area_input"),7)
            #input solid angle
            GL.glActiveTexture(GL.GL_TEXTURE6)
            GL.glBindTexture(GL.GL_TEXTURE_2D,self.SANGLE)
            GL.glUniform1i(GL.glGetUniformLocation(self.prog.prog,"solid_angle_input"),6)
            #input color
            textures=[GL.GL_TEXTURE0,GL.GL_TEXTURE1,GL.GL_TEXTURE2,GL.GL_TEXTURE3,GL.GL_TEXTURE4,GL.GL_TEXTURE5,]
            for id in range(6):
                GL.glActiveTexture(textures[id])
                GL.glBindTexture(GL.GL_TEXTURE_2D,self.TARGET[id])
                GL.glUniform1i(GL.glGetUniformLocation(self.prog.prog,"color_input[%d]"%id),id)
            GL.glUniform1i(GL.glGetUniformLocation(self.prog.prog,"szBuffer"),self.szBuffer)
            GL.glUniform1i(GL.glGetUniformLocation(self.prog.prog,"res"),self.res)
             
            GL.glEnableClientState(GL.GL_VERTEX_ARRAY)
            GL.glVertexPointer(3,GL.GL_FLOAT,0,[0.,0.,0.])
            GL.glDrawArraysInstanced(GL.GL_POINTS,0,1,self.res*self.res)
            GL.glDisableClientState(GL.GL_VERTEX_ARRAY)
        report_GL_error()
        return cleared
          
    '''
        Used internally by render, adjust camera configuration
    ''' 
    @staticmethod 
    def set_view_projection_matrix_cube(zNear, zFar, d, pos, syncGL=True):
        target=[[1.,0.,0.],[-1.,0.,0.],
                [0.,1.,0.],[0.,-1.,0.],
                [0.,0.,1.],[0.,0.,-1.]]
        up=[[0.,-1.,0.],[0.,-1.,0.],
            [0.,0.,1.],[0.,0.,-1.],
            [0.,-1.,0.],[0.,-1.,0.]]
        #setup model view perspective matrix
        MT = Visibility.set_view_matrix(pos+np.array(target[d]), pos, np.array(up[d]), syncGL)
        P = Visibility.set_projection_matrix(90, 1, zNear, zFar, syncGL)
        report_GL_error()
        return MT,P
          
    '''
        Used internally by render, adjust camera configuration
    ''' 
    @staticmethod
    def set_view_matrix(target, pos, up, syncGL=True):
        F = target - pos
        f = ShaderMesh.normalize(F)
        U = ShaderMesh.normalize(up)
        s = ShaderMesh.normalize(np.cross(f, U))
        u = ShaderMesh.normalize(np.cross(s, f))
        M = np.matrix(np.identity(4))
        M[:3,:3] = np.vstack([s,u,-f])
        T = np.matrix([[1.0, 0.0, 0.0, -pos[0]],
                       [0.0, 1.0, 0.0, -pos[1]],
                       [0.0, 0.0, 1.0, -pos[2]],
                       [0.0, 0.0, 0.0, 1.0]])
        MT = M*T
        if syncGL:
            GL.glMatrixMode(GL.GL_MODELVIEW)
            GL.glLoadMatrixf(MT.flatten('F').astype(GL.GLfloat))
        return MT
    
    '''
        Used internally by render, adjust camera configuration
    ''' 
    @staticmethod
    def set_projection_matrix(fov, ar, zNear, zFar, syncGL=True):
        s = 1.0 / math.tan(math.radians(fov)/2.0)
        zNear *= math.cos(math.radians(fov)/2.0)
        
        sx, sy = s / ar, s
        zz = (zFar+zNear) / (zNear-zFar)
        zw = (2*zFar*zNear) / (zNear-zFar)
        P = np.matrix([[sx,0,0,0],
                       [0,sy,0,0],
                       [0,0,zz,zw],
                       [0,0,-1,0]])
        if syncGL:
            GL.glMatrixMode(GL.GL_PROJECTION)
            GL.glLoadMatrixf(P.flatten('F').astype(GL.GLfloat))
        return P
                
    '''
        Read back rendered triangle id (for debug only)
        Output: an image of format GL_RGB(A), with the R/G/B channel stores the sub-parts of a triangle id
        The full triangle id can be recovered by: R*(255*255)+G*255+B
    '''  
    def read_renderbuffer(self,d):
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.TARGET[d])
        str = GL.glGetTexImage(GL.GL_TEXTURE_2D, 0, GL.GL_RGB, GL.GL_UNSIGNED_BYTE)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        data = np.fromstring(str, dtype=np.uint8)
        return data.reshape((self.res,self.res,3))
             
    '''
        Read back visibility
        Output: the linear buffer containing solid angle per triangle
        Output: the linear buffer containing solid angle / triangle area
    '''  
    def read_accumulatebuffer(self):
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.ACCUM)
        str = GL.glGetTexImage(GL.GL_TEXTURE_2D, 0, GL.GL_RGB, GL.GL_FLOAT)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        data = np.fromstring(str, dtype=np.float32)
        data=data.reshape((self.szBuffer*self.szBuffer,3))
        return data[:len(self.mesh.mesh.faces),0],data[:len(self.mesh.mesh.faces),1]
      
    @staticmethod
    def debug_render_shader(path, res, pt, write=False, robot=None, bulbId=None):
        vis1=Visibility(path, res, True)
        vis1.set_robot(robot, bulbId)
        acc1,acca1=vis1.render(len(vis1.mesh.mesh.faces)//2, None, pt)
        dat1=[vis1.read_renderbuffer(d) for d in range(6)]
        print('GPU:',acc1.sum(),acca1.sum())
        del vis1
        
        vis2=Visibility(path, res, False)
        vis2.set_robot(robot, bulbId)
        acc2,acca2=vis2.render(len(vis2.mesh.mesh.faces)//2, None, pt, accumCPU=True)
        dat2=[vis2.read_renderbuffer(d) for d in range(6)]
        print('CPU:',acc2.sum(),acca2.sum())
        del vis2
        
        print("Err:",np.amax(acc1-acc2))
        
        #compare
        for i in range(6):
            assert (dat1[i]==dat2[i]).all()
            
    @staticmethod
    def debug_render_framerate(path, res, ptRange, nrTrial=1000, nrLightSample=1):
        FPS=[]
        import random,time
        vis=Visibility(path, res, True)
        for i in range(nrTrial):
            poses=[[random.uniform(ptRange[0],ptRange[1]) for i in range(3)] for l in range(nrLightSample)]
            a=time.perf_counter()
            vis.render(len(vis.mesh.mesh.faces)//2, None, poses)
            b=time.perf_counter()
            FPS.append(1/(b-a))
            print("Frame%d-FPS: %f"%(i,FPS[-1]))
        del vis
            
        if not os.path.exists("output"):
            os.mkdir("output")
        import matplotlib.pyplot as plt
        plt.hist(FPS,10)
        name=os.path.basename(path).split('.')[0]
        plt.xlabel('Instantaneous FPS at resolution=%d, mesh=%s, #LightSample=%d'%(res,name,nrLightSample))
        plt.ylabel('#Frames')
        plt.grid(True)
        plt.savefig('output/%d_%s_%d.pdf'%(res,name,nrLightSample))
        plt.show()
            
if __name__=='__main__':
    Visibility.debug_render_shader("../data/visibility_tests/sphere.obj", 128, [24,0,0], robot='../data/ur5e.rob', bulbId=8)
    Visibility.debug_render_framerate('../data/visibility_tests/data/room100k.obj', 1024, [0,10000], nrLightSample=1)
    Visibility.debug_render_framerate('../data/visibility_tests/data/room100k.obj', 1024, [0,10000], nrLightSample=4)
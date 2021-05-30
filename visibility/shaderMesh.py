import math,glfw,klampt,trimesh as tm,numpy as np
from visibility.shaderProgram import ShaderProgram,report_GL_error
from OpenGL import GL

class ShaderMesh:
    '''
        This class takes a 3D triangle mesh and creates GPU buffers for fast rendering
        Input: path can be 1) a string of file location, a klampt.Geometry3D object, or a trimesh.Trimesh object
    '''
    def __init__(self, path):
        if isinstance(path,str):
            self.mesh=tm.exchange.load.load(path)
        elif isinstance(path,klampt.Geometry3D):
            g=path.getTriangleMesh()
            vss=[tuple(g.vertices[i*3:i*3+3]) for i in range(len(g.vertices)//3)]
            iss=[tuple(g.indices[i*3:i*3+3]) for i in range(len(g.indices)//3)]
            self.mesh=tm.Trimesh(vss,iss)
        else: self.mesh=path
        
    '''
        Output: mass center of the watertight mesh
    '''
    def centroid(self):
        c=self.mesh.centroid
        return [c[0],c[1],c[2]]
        
    '''
        This is a pre-processnig function that create internal data for rendering without any special effect
    '''
    def init_render(self):
        self.useShader=False
        
        #create vertex array
        self.verts = []
        for V in self.mesh.vertices:
            for v in V:
                self.verts.append(v)
        self.verts=np.array(self.verts).astype(GL.GLfloat)
        
        #create index array
        self.indices = []
        for F in self.mesh.faces:
            for f in F:
                self.indices.append(f)
        self.indices=np.array(self.indices).astype(GL.GLuint)
        
        #create normal array
        self.normals = []
        for N in self.mesh.vertex_normals:
            for n in N:
                self.normals.append(n)
        self.normals=np.array(self.normals).astype(GL.GLfloat)
           
    '''
        This is a pre-processnig function that create internal data for rendering in the mode of GL_TRIANGLES_ADJACENCY
    ''' 
    def init_render_silhouette(self):
        self.adjs={}
        for a,b in self.mesh.face_adjacency:
            for i in self.mesh.faces[a]:
                if i not in self.mesh.faces[b]:
                    ia=i
            for i in self.mesh.faces[b]:
                if i not in self.mesh.faces[a]:
                    ib=i
            self.adjs[(b,ShaderMesh.id(self.mesh.faces[b],ib))]=ia
            self.adjs[(a,ShaderMesh.id(self.mesh.faces[a],ia))]=ib
            
        fid=0
        self.indices_adjacency=[]
        for F in self.mesh.faces:
            if (fid,0) in self.adjs and (fid,1) in self.adjs and (fid,2) in self.adjs:
                self.indices_adjacency+=[F[0],self.adjs[(fid,2)],F[1],self.adjs[(fid,0)],F[2],self.adjs[(fid,1)]]
            fid+=1
        self.indices_adjacency=np.array(self.indices_adjacency).astype(GL.GLuint)
            
    '''
        This is a pre-processnig function that create internal data for rendering triangle indices (used by visibility)
        This will render an image of format GL_RGB(A), with the R/G/B channel stores the sub-parts of a triangle id
        The full triangle id can be recovered by: R*(255*255)+G*255+B
        Input: whether triangle ids should be computed on GPU using shaders?
    ''' 
    def init_visibility(self, useShader=True):
        self.useShader=useShader
        if self.useShader:
            vert='''
                #version 410 compatibility
                void main()
                {
                    gl_Position=gl_ModelViewProjectionMatrix*gl_Vertex; 
                }
            '''
            frag='''
                #version 410 compatibility
                in int gl_PrimitiveID;
                uniform int id0,id1;
                void main()
                {
                    if(gl_PrimitiveID>=id0 && gl_PrimitiveID<id1) {
                        int f=gl_PrimitiveID+1-id0;
                        int r=f/(255*255);
                        int gb=f%(255*255);
                        int g=gb/255;
                        int b=gb%255;
                        gl_FragColor=vec4(float(r)/255.0,float(g)/255.0,float(b)/255.0,1);
                    } else {
                        gl_FragColor=vec4(0.0,0.0,0.0,1.0);
                    }
                } 
            '''
            self.prog=ShaderProgram(vert=vert,frag=frag)
            #create vertex array
            self.verts = []
            for V in self.mesh.vertices:
                for v in V:
                    self.verts.append(v)
            self.verts=np.array(self.verts).astype(GL.GLfloat)
            
            #create index array
            self.indices = []
            for F in self.mesh.faces:
                for f in F:
                    self.indices.append(f)
            self.indices=np.array(self.indices).astype(GL.GLuint)
        else:
            #create vertex array
            self.verts = []
            for f in self.mesh.faces:
                for d in f:
                    for v in self.mesh.vertices[d]:
                        self.verts.append(v)
            self.verts=np.array(self.verts).astype(GL.GLfloat)
            
    '''
        Delete shader object
    ''' 
    def __del__(self):
        if hasattr(self,'useShader') and self.useShader:
            del self.prog
    
    '''
        This function computes the best zNear,zFar to be used by OpenGL's gluPerspective function
        Input: camera position
        Output: zNear,zFar
    ''' 
    def compute_zNear_zFar(self, pos):
        #compute zFar approximately
        zFar = 0.
        for d in range(3):
            vmin = self.mesh.bounds[0][d]
            vmax = self.mesh.bounds[1][d]
            distD = max(abs(pos[d]-vmin),abs(pos[d]-vmax))
            zFar += distD*distD
        zFar = math.sqrt(zFar)
        
        #compute zNear via BVH
        _,dist,_=tm.proximity.closest_point(self.mesh,np.reshape(pos,(1,3)))
        zNear = dist[0]
        
        return zNear, zFar

    '''
        After calling init_render, call this function to actually perform the rendering
    '''
    def draw_render(self):
        GL.glEnableClientState(GL.GL_VERTEX_ARRAY)
        GL.glEnableClientState(GL.GL_NORMAL_ARRAY)
        GL.glVertexPointer(3, GL.GL_FLOAT, 0, self.verts)
        GL.glNormalPointer(GL.GL_FLOAT, 0, self.normals)
        GL.glDrawElements(GL.GL_TRIANGLES, len(self.indices), GL. GL_UNSIGNED_INT, self.indices)
        GL.glDisableClientState(GL.GL_VERTEX_ARRAY)
        GL.glDisableClientState(GL.GL_NORMAL_ARRAY)
        report_GL_error()

    '''
        After calling init_render_silhouette, call this function to actually perform the rendering
    '''
    def draw_render_silhouette(self):
        GL.glEnableClientState(GL.GL_VERTEX_ARRAY)
        GL.glEnableClientState(GL.GL_NORMAL_ARRAY)
        GL.glVertexPointer(3, GL.GL_FLOAT, 0, self.verts)
        GL.glNormalPointer(GL.GL_FLOAT, 0, self.normals)
        GL.glDrawElements(GL.GL_TRIANGLES_ADJACENCY, len(self.indices_adjacency), GL. GL_UNSIGNED_INT, self.indices_adjacency)
        GL.glDisableClientState(GL.GL_VERTEX_ARRAY)
        GL.glDisableClientState(GL.GL_NORMAL_ARRAY)
        report_GL_error()

    '''
        After calling init_visibility, call this function to actually perform the rendering
        Input: choose to only render triangle indices in range id0<=id<id1
        Input: pt is currently unused, just put any 3 tuple
    '''
    def draw_visibility(self, id0, id1, pt):
        if id0==None:
            id0=0
        if id1==None:    
            id1=len(self.mesh.faces)
        
        if self.useShader:
            with self.prog:
                GL.glUniform1i(GL.glGetUniformLocation(self.prog.prog,"id0"),id0)
                GL.glUniform1i(GL.glGetUniformLocation(self.prog.prog,"id1"),id1)
                GL.glUniform4f(GL.glGetUniformLocation(self.prog.prog,"pt"),pt[0],pt[1],pt[2],1.0)
                
                GL.glEnableClientState(GL.GL_VERTEX_ARRAY)
                GL.glVertexPointer(3, GL.GL_FLOAT, 0, self.verts)
                GL.glDrawElements(GL.GL_TRIANGLES, len(self.indices), GL. GL_UNSIGNED_INT, self.indices)
                GL.glDisableClientState(GL.GL_VERTEX_ARRAY)
        else:
            #create color array
            self.colors = []
            for fid,f in enumerate(self.mesh.faces):
                if fid>=id0 and fid<id1:
                    r,g,b=ShaderMesh.convert_to_color(fid-id0)
                else: r,g,b=0.0,0.0,0.0
                for d in f:
                    self.colors.append(r)
                    self.colors.append(g)
                    self.colors.append(b)
            self.colors=np.array(self.colors).astype(GL.GLubyte)
            
            GL.glEnableClientState(GL.GL_VERTEX_ARRAY)
            GL.glVertexPointer(3, GL.GL_FLOAT, 0, self.verts)
            GL.glEnableClientState(GL.GL_COLOR_ARRAY)
            GL.glColorPointer(3, GL.GL_UNSIGNED_BYTE, 0, self.colors)
            GL.glDrawArrays(GL.GL_TRIANGLES, 0, self.verts.shape[0]//3)
            GL.glDisableClientState(GL.GL_VERTEX_ARRAY)
            GL.glDisableClientState(GL.GL_COLOR_ARRAY)
        report_GL_error()
          
    '''
        Output: a list of triangle areas
    '''
    def area(self):
        if not hasattr(self,"areas"):
            ret=[]
            for F in self.mesh.faces:
                u1=np.array([i for i in self.mesh.vertices[F[0]]])
                u2=np.array([i for i in self.mesh.vertices[F[1]]])
                u3=np.array([i for i in self.mesh.vertices[F[2]]])
                ret.append(ShaderMesh.area_of_triangle(u1,u2,u3))
            self.areas=np.array(ret)
        return self.areas
       
    @staticmethod
    def id(arr,val):
        for i in range(3):
            if arr[i]==val:
                return i
        assert False
       
    @staticmethod    
    def convert_to_color(f):
        f+=1
        r=f//(255*255)
        gb=f%(255*255)
        g=gb//255
        b=gb%255
        return r,g,b
    
    @staticmethod
    def normalize(v):
        m = math.sqrt(np.sum(v ** 2))
        if m == 0:
            return v
        return v / m
    
    @staticmethod
    def area_of_triangle(u1, u2, u3):
        d1=u2-u1
        d2=u3-u1
        return np.linalg.norm(np.cross(d1,d2))/2
    
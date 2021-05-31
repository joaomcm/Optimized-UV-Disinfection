from OpenGL import GL

def report_GL_error():
    ret=GL.glGetError()
    if ret!=GL.GL_NO_ERROR:
        raise RuntimeError("OpenGL Error: %s"%str(ret))

class ShaderProgram:
    '''
        This class is a shader wrapper
        Input: optionally pass vertex, geometry, fragment, compute shaders as a str, None means the use of default system pipeline
        Throw: RuntimeError if there are any compile time or link error
    '''
    def __init__(self,vert=None,geom=None,frag=None,compute=None):
        #shader
        self.vert=vert
        self.geom=geom
        self.frag=frag
        self.compute=compute
        self.shaders=[]
        for code,type in zip([vert,geom,frag,compute],[GL.GL_VERTEX_SHADER,GL.GL_GEOMETRY_SHADER,GL.GL_FRAGMENT_SHADER,GL.GL_COMPUTE_SHADER]):
            if code is not None:
                self.shaders.append(GL.glCreateShader(type))
                GL.glShaderSource(self.shaders[-1],code)
                GL.glCompileShader(self.shaders[-1])
                compiled=GL.glGetShaderiv(self.shaders[-1],GL.GL_COMPILE_STATUS)
                if not compiled:
                    info=GL.glGetShaderInfoLog(self.shaders[-1])
                    GL.glDeleteShader(self.shaders[-1])
                    raise RuntimeError("Shader Compile Error: %s"%info)
        
        #program
        self.prog=GL.glCreateProgram()
        for shader in self.shaders:
            GL.glAttachShader(self.prog,shader)
        GL.glLinkProgram(self.prog)
        linked=GL.glGetProgramiv(self.prog,GL.GL_LINK_STATUS)
        if not linked:
            info=GL.glGetProgramInfoLog(self.prog)
            GL.glDeleteProgram(self.prog)
            for shader in self.shaders:
                GL.glDeleteShader(shader)
            raise RuntimeError("Program Link Error: %s"%info)
        
        #finish
        for shader in self.shaders:
            GL.glDetachShader(self.prog,shader)
        report_GL_error()
    
    def __del__(self):
        GL.glDeleteProgram(self.prog)
        for shader in self.shaders:
            GL.glDeleteShader(shader)
        
    def __enter__(self):
        GL.glUseProgram(self.prog)
        report_GL_error()
    
    def __exit__(self, exception_type, exception_value, exception_traceback):
        GL.glUseProgram(0)
        report_GL_error()
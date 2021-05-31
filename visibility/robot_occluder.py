import klampt
from .shader_mesh import ShaderMesh
from .shader_program import ShaderProgram
from .visualizer import GLVisualizer
from klampt.math import se3

class Occluder:
    '''
        This class stores a robot used as visibility occluder
        Input: a klampt.RobotModel object
        Input: a link id on the robot that represents the bulb
    '''
    def __init__(self,robot,bulbId):
        if isinstance(robot,str):
            self.world=klampt.WorldModel()
            self.world.loadRobot(robot)
            robot=self.world.robot(0)
        
        self.robot=robot
        self.bulbId=bulbId
        self.localPos= ShaderMesh(robot.link(bulbId).geometry()).centroid()
        self.frag='''
            #version 410 compatibility
            void main()
            {
                gl_FragColor=vec4(0.0,0.0,0.0,1.0);
            }
        '''
        
    '''
        Output: the current light position
    '''
    def get_light_positions(self):
        t=self.robot.link(self.bulbId).getTransform()
        return [se3.apply(t,self.localPos)]
        
    '''
        Convenient function for moving the light to a specified position using inverse kinematics
        Input: desired light position
    '''
    def set_config(self,pt):
        if len(pt)==3:
            #print(self.robot.getConfig())
            goal=klampt.model.ik.objective(self.robot.link(self.bulbId),local=self.localPos,world=pt)
            klampt.model.ik.solve(goal)
            #print(self.robot.getConfig())
        else: 
            self.robot.setConfig(pt)
        
    '''
        Render the robot as a black object to flag the visibility computer that this is an occluder
    '''
    def drawGL(self):
        if not hasattr(self,'prog'):
            self.prog=ShaderProgram(frag=self.frag)
        with self.prog:
            for i in range(self.robot.numLinks()):
                if i!=self.bulbId:
                    self.robot.link(i).drawWorldGL(False)
        
class CustomGLVisualizer(GLVisualizer):
    def __init__(self,world,bulbId):
        GLVisualizer.__init__(self,world)
        self.occluder=Occluder(self.world.robot(0),bulbId)
    
    def display(self):
        self.occluder.drawGL()
            
    def keyboardfunc(self,c,x,y):
        GLVisualizer.keyboardfunc(self,c,x,y)
        if c=='1':
            self.occluder.bulbId=(self.occluder.bulbId+self.world.robot(0).numLinks()-1)%self.world.robot(0).numLinks()
            print('bulbId=%d'%self.bulbId)
        elif c=='2':
            self.occluder.bulbId=(self.occluder.bulbId+1)%self.world.robot(0).numLinks()
            print('bulbId=%d'%self.bulbId)
        elif c=='3':
            l=self.occluder.get_light_position()
            print('bulbId=%d, centroid=(%f,%f,%f)'%(self.occluder.bulbId,l[0],l[1],l[2]))
        elif c=='4':
            self.occluder.bulbId=self.world.robot(0).numLinks()
    
    def idle(self):
        l=self.occluder.get_light_positions()[0]
        print('bulbId=%d, centroid=(%f,%f,%f)'%(self.occluder.bulbId,l[0],l[1],l[2]))
        
if __name__=='__main__':
    world=klampt.WorldModel()
    world.loadRobot('../data/ur5e.rob')
    CustomGLVisualizer(world,8).run()
    
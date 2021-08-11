def make(robotfile,world,tempname="disinfection_robot.rob",debug=True):
    """Converts the given fixed-base robot file into a moving base robot
    and loads it into the given world.

    Args:
        robotfile (str): the name of a fixed-base robot file to load
        world (WorldModel): a world that will contain the new robot
        tempname (str, optional): a name of a temporary file containing
            the moving-base robot
        debug (bool, optional): if True, the robot file named by
            ``tempname`` is not removed from disk.

    Returns:
        (RobotModel): the loaded robot, stored in ``world``.
    """
    _template_ = """### Boilerplate kinematics of a drivable floating (translating and rotating) cube with a robot hand mounted on it
    TParent 1 0 0   0 1 0   0 0 1   0 0 0  \\
    1 0 0   0 1 0   0 0 1   0 0 0  \\
    1 0 0   0 1 0   0 0 1   0 0 0  
    parents -1 0 1
    axis 1 0 0   0 1 0    0 0 1
    jointtype p p p 
    qMin -100 -100 -100
    qMax  100  100  100 
    q 0 0 0
    links "tx" "ty" "tz"
    geometry   ""   "" "primitives/scaled_cylinder.off"
    geomscale 1 1 1
    mass       0.1 0.1 0.1
    com 0 0 0   0 0 0   0 0 0
    inertia 0.001 0 0 0 0.001 0 0 0 0.001 \\
       0.001 0 0 0 0.001 0 0 0 0.001 \\
       0.001 0 0 0 0.001 0 0 0 0.001 
    torqueMax  500 500 500
    accMax     4 4 4
    velMax     2 2 2

    joint normal 0
    joint normal 1
    joint normal 2

    driver normal 0 
    driver normal 1
    driver normal 2

    servoP 5000 5000 5000 
    servoI 10 10 10 
    servoD 100 100 100 
    viscousFriction 50 50 50 
    dryFriction 1 1 1

    property sensors <sensors><ForceTorqueSensor name="base_force" link="2" hasForce="1 1 1" hasTorque="1 1 1" /></sensors>
    mount 2 "%s" 1 0 0   0 1 0   0 0 1   0 0 0.37 as "%s"
    """

    robotname = os.path.splitext(os.path.basename(robotfile))[0]
    f = open(tempname,'w')
    f.write(_template_ % (robotfile,robotname))
    f.close()
    world.loadElement(tempname)
    robot = world.robot(world.numRobots()-1)
    #set torques
    mass = sum(robot.link(i).getMass().mass for i in range(robot.numLinks()))
    inertia = 0.0
    for i in range(robot.numLinks()):
        m = robot.link(i).getMass()
        inertia += (vectorops.normSquared(m.com)*m.mass + max(m.inertia))
    tmax = robot.getTorqueLimits()
    tmax[0] = tmax[1] = tmax[2] = mass*9.8*5
    tmax[3] = tmax[4] = tmax[5] = inertia*9.8*5
    robot.setName("moving-base["+robotname+"]")
    robot.setTorqueLimits(tmax)
    if debug:
        robot.saveFile(tempname)
    else:
        os.remove(tempname)
    return robot

def setup_robot_and_light(robotfile = './primitives/ur5e.rob',
                                     mesh_file = './full_detail_hospital_cad_meters.obj'):
    world = WorldModel()
    robot = make(robotfile,world)



    res = world.loadElement(mesh_file)
    print(res)
    world.terrain(0).geometry().setCollisionMargin(0.01)
    collider = WorldCollider(world)
    list(collider.collisions())
    
    cfig = robot.getConfig()
    terrain = world.terrain(0)
    lamp = robot.link(11)
    cfig[2] = 0.07
    robot.setConfig(cfig)
    robot.link(11).appearance().setColor(210/255,128/255,240/255,1)

    world.saveFile('disinfection.xml')

    
    return world,robot,lamp,collider

def determine_reachable_points_robot(sampling_places,world,robot,lamp,collider,show_vis = False,):
    # myrob = resource.get('/home/motion/Klampt-examples/data/robots/tx90ball.rob')



    def collisionChecker():
        if(list(collider.collisions())!= []):
            return False

        return True

    if(show_vis):
        vis.add('world',world)
        vis.show()
    
    reachable = []
    for place in tqdm(sampling_places):
        goal = place.tolist()
        obj = ik.objective(lamp,local = [0,0,0], world = goal)
        reachable.append(ik.solve_global(obj,numRestarts = 100,activeDofs = [0,1,4,5,6,7,8,9],feasibilityCheck = collisionChecker))
#         time.sleep(1)
#         if(not reachable[-1]):
#             time.sleep(1)
#             print('unreachable!')
#     print(lamp.getTransform())
    return reachable
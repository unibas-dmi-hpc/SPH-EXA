from paraview.simple import TrivialProducer

reader = TrivialProducer(registrationName="grid")

def catalyst_initialize():
    print("catalyst_initialize()")

def catalyst_execute(params):
    if not (params.timestep % 1):
        print("catalyst_execute: particles' Density range", reader.PointData["Density"].GetRange(), " at timestep:", params.timestep)

def catalyst_finalize():
    print("catalyst_finalize()")

from paraview import catalyst
options = catalyst.Options()
options.GlobalTrigger = 'TimeStep'
options.EnableCatalystLive = 0
options.CatalystLiveTrigger = 'TimeStep'
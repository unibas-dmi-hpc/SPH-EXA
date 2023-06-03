

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()


# get the material library
materialLibrary1 = GetMaterialLibrary()

# Create a new 'Render View'
renderView1 = CreateView('RenderView')
renderView1.ViewSize = [1024,1024]
renderView1.AxesGrid = 'GridAxes3DActor'
renderView1.CenterOfRotation = [3.9400288337487765e-08, -9.984030726328808e-07, 1.4839057228266395e-06]
renderView1.StereoType = 'Crystal Eyes'
renderView1.CameraPosition = [-2.3042363258838767, 1.4345831891542806, 2.0396833813115496]
renderView1.CameraFocalPoint = [3.940028833748751e-08, -9.98403072632882e-07, 1.4839057228266399e-06]
renderView1.CameraViewUp = [0.31184497002486106, 0.906330275196784, -0.2851633688465534]
renderView1.CameraFocalDisk = 1.0
renderView1.CameraParallelScale = 0.8796674210602298
renderView1.BackEnd = 'OSPRay raycaster'
renderView1.OSPRayMaterialLibrary = materialLibrary1


reader = TrivialProducer(registrationName="grid")

# create a new 'Outline Corners'
outlineCorners1 = OutlineCorners(registrationName='OutlineCorners1', Input=reader)

# create a new 'SPH Plane Interpolator'
Yplane = SPHPlaneInterpolator(registrationName='Yplane', Input=reader,
    Source='Bounded Plane')
Yplane.DensityArray = 'Density'
Yplane.MassArray = 'Mass'
Yplane.CutoffArray = 'None'
Yplane.ExcludedArrays = ['Mass', 'Pressure']
Yplane.Kernel = 'SPHQuinticKernel'
Yplane.Locator = 'Static Point Locator'
Yplane.Kernel.SpatialStep = 0.1
Yplane.Source.BoundingBox = [-0.5, 0.5, -0.5, 0.5, -0.5, 0.5]
Yplane.Source.Resolution = 400
Yplane.Source.Center = [0., -0., 0.]
Yplane.Source.Normal = [0.0, 1.0, 0.0]

# create a new 'SPH Plane Interpolator'
Zplane = SPHPlaneInterpolator(registrationName='Zplane', Input=reader,
    Source='Bounded Plane')
Zplane.DensityArray = 'Density'
Zplane.MassArray = 'Mass'
Zplane.CutoffArray = 'None'
Zplane.ExcludedArrays = ['Mass', 'Pressure']
Zplane.Kernel = 'SPHQuinticKernel'
Zplane.Locator = 'Static Point Locator'
Zplane.Kernel.SpatialStep = 0.1
Zplane.Source.BoundingBox = [-0.5, 0.5, -0.5, 0.5, -0.5, 0.5]
Zplane.Source.Resolution = 400
Zplane.Source.Center = [0., -0., 0.]
Zplane.Source.Normal = [0.0, 0.0, 1.0]

# create a new 'SPH Plane Interpolator'
Xplane = SPHPlaneInterpolator(registrationName='Xplane', Input=reader,
    Source='Bounded Plane')
Xplane.DensityArray = 'Density'
Xplane.MassArray = 'Mass'
Xplane.CutoffArray = 'None'
Xplane.ExcludedArrays = ['Mass', 'Pressure']
Xplane.Kernel = 'SPHQuinticKernel'
Xplane.Locator = 'Static Point Locator'
Xplane.Kernel.SpatialStep = 0.1
Xplane.Source.BoundingBox = [-0.5, 0.5, -0.5, 0.5, -0.5, 0.5]
Xplane.Source.Resolution = 400
Xplane.Source.Center = [0., -0., 0.]
Xplane.Source.Normal = [1.0, 0.0, 0.0]

# ----------------------------------------------------------------
# setup the visualization in view 'renderView1'
# ----------------------------------------------------------------

# show data from Zplane
ZplaneDisplay = Show(Zplane, renderView1, 'GeometryRepresentation')

# get color transfer function/color map for 'Density'
densityLUT = GetColorTransferFunction('Density')
densityLUT.RGBPoints = [0.06410533205959397, 0.231373, 0.298039, 0.752941, 0.7290025733056157, 0.865003, 0.865003, 0.865003, 1.3938998145516373, 0.705882, 0.0156863, 0.14902]
densityLUT.ScalarRangeInitialized = 1.0

# trace defaults for the display properties.
ZplaneDisplay.Representation = 'Surface'
ZplaneDisplay.ColorArrayName = ['POINTS', 'Density']
ZplaneDisplay.LookupTable = densityLUT
ZplaneDisplay.OSPRayScaleArray = 'Density'
ZplaneDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
ZplaneDisplay.SelectOrientationVectors = 'None'
ZplaneDisplay.ScaleFactor = 0.0999997526407242
ZplaneDisplay.SelectScaleArray = 'Density'
ZplaneDisplay.GlyphType = 'Arrow'
ZplaneDisplay.GlyphTableIndexArray = 'Density'
ZplaneDisplay.GaussianRadius = 0.004999987632036209
ZplaneDisplay.SetScaleArray = ['POINTS', 'Density']
ZplaneDisplay.ScaleTransferFunction = 'PiecewiseFunction'
ZplaneDisplay.OpacityArray = ['POINTS', 'Density']
ZplaneDisplay.OpacityTransferFunction = 'PiecewiseFunction'
ZplaneDisplay.DataAxesGrid = 'GridAxesRepresentation'
ZplaneDisplay.PolarAxes = 'PolarAxesRepresentation'

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
ZplaneDisplay.ScaleTransferFunction.Points = [0.06410533205959397, 0.0, 0.5, 0.0, 1.3955178683033722, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
ZplaneDisplay.OpacityTransferFunction.Points = [0.06410533205959397, 0.0, 0.5, 0.0, 1.3955178683033722, 1.0, 0.5, 0.0]

# show data from outlineCorners1
outlineCorners1Display = Show(outlineCorners1, renderView1, 'GeometryRepresentation')

# trace defaults for the display properties.
outlineCorners1Display.Representation = 'Surface'
outlineCorners1Display.ColorArrayName = [None, '']
outlineCorners1Display.OSPRayScaleFunction = 'PiecewiseFunction'
outlineCorners1Display.SelectOrientationVectors = 'None'
outlineCorners1Display.ScaleFactor = 0.0999997526407242
outlineCorners1Display.SelectScaleArray = 'None'
outlineCorners1Display.GlyphType = 'Arrow'
outlineCorners1Display.GlyphTableIndexArray = 'None'
outlineCorners1Display.GaussianRadius = 0.004999987632036209
outlineCorners1Display.SetScaleArray = [None, '']
outlineCorners1Display.ScaleTransferFunction = 'PiecewiseFunction'
outlineCorners1Display.OpacityArray = [None, '']
outlineCorners1Display.OpacityTransferFunction = 'PiecewiseFunction'
outlineCorners1Display.DataAxesGrid = 'GridAxesRepresentation'
outlineCorners1Display.PolarAxes = 'PolarAxesRepresentation'
# show data from Yplane
YplaneDisplay = Show(Yplane, renderView1, 'GeometryRepresentation')

# trace defaults for the display properties.
YplaneDisplay.Representation = 'Surface'
YplaneDisplay.ColorArrayName = ['POINTS', 'Density']
YplaneDisplay.LookupTable = densityLUT
YplaneDisplay.OSPRayScaleArray = 'Density'
YplaneDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
YplaneDisplay.SelectOrientationVectors = 'None'
YplaneDisplay.ScaleFactor = 0.0999997526407242
YplaneDisplay.SelectScaleArray = 'Density'
YplaneDisplay.GlyphType = 'Arrow'
YplaneDisplay.GlyphTableIndexArray = 'Density'
YplaneDisplay.GaussianRadius = 0.004999987632036209
YplaneDisplay.SetScaleArray = ['POINTS', 'Density']
YplaneDisplay.ScaleTransferFunction = 'PiecewiseFunction'
YplaneDisplay.OpacityArray = ['POINTS', 'Density']
YplaneDisplay.OpacityTransferFunction = 'PiecewiseFunction'
YplaneDisplay.DataAxesGrid = 'GridAxesRepresentation'
YplaneDisplay.PolarAxes = 'PolarAxesRepresentation'


# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
YplaneDisplay.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.1757813367477812e-38, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
YplaneDisplay.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.1757813367477812e-38, 1.0, 0.5, 0.0]

# show data from Xplane
XplaneDisplay = Show(Xplane, renderView1, 'GeometryRepresentation')

# trace defaults for the display properties.
XplaneDisplay.Representation = 'Surface'
XplaneDisplay.ColorArrayName = ['POINTS', 'Density']
XplaneDisplay.LookupTable = densityLUT
XplaneDisplay.OSPRayScaleArray = 'Density'
XplaneDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
XplaneDisplay.SelectOrientationVectors = 'None'
XplaneDisplay.ScaleFactor = 0.0999997526407242
XplaneDisplay.SelectScaleArray = 'Density'
XplaneDisplay.GlyphType = 'Arrow'
XplaneDisplay.GlyphTableIndexArray = 'Density'
XplaneDisplay.GaussianRadius = 0.004999987632036209
XplaneDisplay.SetScaleArray = ['POINTS', 'Density']
XplaneDisplay.ScaleTransferFunction = 'PiecewiseFunction'
XplaneDisplay.OpacityArray = ['POINTS', 'Density']
XplaneDisplay.OpacityTransferFunction = 'PiecewiseFunction'
XplaneDisplay.DataAxesGrid = 'GridAxesRepresentation'
XplaneDisplay.PolarAxes = 'PolarAxesRepresentation'
# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
XplaneDisplay.ScaleTransferFunction.Points = [0.06410533205959397, 0.0, 0.5, 0.0, 1.3938998145516373, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
XplaneDisplay.OpacityTransferFunction.Points = [0.06410533205959397, 0.0, 0.5, 0.0, 1.3938998145516373, 1.0, 0.5, 0.0]

# setup the color legend parameters for each legend in this view

# get color legend/bar for densityLUT in view renderView1
densityLUTColorBar = GetScalarBar(densityLUT, renderView1)
densityLUTColorBar.Title = 'Density'
densityLUTColorBar.ComponentTitle = ''

# set color bar visibility
densityLUTColorBar.Visibility = 1

# show color legend
ZplaneDisplay.SetScalarBarVisibility(renderView1, True)

# show color legend
YplaneDisplay.SetScalarBarVisibility(renderView1, True)

# show color legend
XplaneDisplay.SetScalarBarVisibility(renderView1, True)

# ----------------------------------------------------------------
# setup color maps and opacity mapes used in the visualization
# note: the Get..() functions create a new object, if needed
# ----------------------------------------------------------------

# get opacity transfer function/opacity map for 'Density'
densityPWF = GetOpacityTransferFunction('Density')
densityPWF.Points = [0.06410533205959397, 0.0, 0.5, 0.0, 1.3938998145516373, 1.0, 0.5, 0.0]
densityPWF.ScalarRangeInitialized = 1

# ----------------------------------------------------------------
# restore active source
SetActiveSource(Xplane)

pNG1 = CreateExtractor('PNG', renderView1, registrationName='PNG1')
pNG1.Trigger = 'TimeStep'
pNG1.Trigger.Frequency = 10
#pNG1.Writer.FileName = 'test1_{timestep:06d}{camera}.png'
pNG1.Writer.FileName = 'SedovPlanes_{timestep:06d}{camera}.png'
pNG1.Writer.ImageResolution = [1024,1024]
pNG1.Writer.Format = 'PNG'

# Catalyst options
from paraview import catalyst
options = catalyst.Options()
options.GlobalTrigger = 'TimeStep'
#options.EnableCatalystLive = 1
options.CatalystLiveTrigger = 'TimeStep'
options.ExtractsOutputDirectory = '/scratch/snx3000/jfavre/sedov'

# ------------------------------------------------------------------------------
if __name__ == '__main__':
    from paraview.simple import SaveExtractsUsingCatalystOptions
    # Code for non in-situ environments; if executing in post-processing
    # i.e. non-Catalyst mode, let's generate extracts using Catalyst options
    SaveExtractsUsingCatalystOptions(options)

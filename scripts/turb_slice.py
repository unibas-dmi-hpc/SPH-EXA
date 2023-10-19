# trace generated using paraview version 5.11.1
#import paraview
#paraview.compatibility.major = 5
#paraview.compatibility.minor = 11

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# create a new 'H5PartReader'
demo_turbulence_150h5 = H5PartReader(registrationName='demo_turbulence_150.h5', FileName='/scratch/snx3000/sebkelle/chkp.avc.final.h5')
demo_turbulence_150h5.Xarray = 'x'
demo_turbulence_150h5.Yarray = 'y'
demo_turbulence_150h5.Zarray = 'z'
demo_turbulence_150h5.PointArrays = ['alpha', 'du_m1', 'h', 'm', 'temp', 'vx', 'vy', 'vz', 'x', 'x_m1', 'y', 'y_m1', 'z', 'z_m1']

# get animation scene
animationScene1 = GetAnimationScene()

# get the time-keeper
timeKeeper1 = GetTimeKeeper()

# update animation scene based on data timesteps
animationScene1.UpdateAnimationUsingDataTimeSteps()

# Properties modified on demo_turbulence_150h5
demo_turbulence_150h5.PointArrays = ['vx', 'x', 'y', 'z']

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')

# show data in view
demo_turbulence_150h5Display = Show(demo_turbulence_150h5, renderView1, 'GeometryRepresentation')

# get color transfer function/color map for 'vx'
vxLUT = GetColorTransferFunction('vx')

# trace defaults for the display properties.
demo_turbulence_150h5Display.Representation = 'Surface'
demo_turbulence_150h5Display.ColorArrayName = ['POINTS', 'vx']
demo_turbulence_150h5Display.LookupTable = vxLUT
demo_turbulence_150h5Display.SelectTCoordArray = 'None'
demo_turbulence_150h5Display.SelectNormalArray = 'None'
demo_turbulence_150h5Display.SelectTangentArray = 'None'
demo_turbulence_150h5Display.OSPRayScaleArray = 'vx'
demo_turbulence_150h5Display.OSPRayScaleFunction = 'PiecewiseFunction'
demo_turbulence_150h5Display.SelectOrientationVectors = 'None'
demo_turbulence_150h5Display.ScaleFactor = 0.09999994716636358
demo_turbulence_150h5Display.SelectScaleArray = 'vx'
demo_turbulence_150h5Display.GlyphType = 'Arrow'
demo_turbulence_150h5Display.GlyphTableIndexArray = 'vx'
demo_turbulence_150h5Display.GaussianRadius = 0.004999997358318179
demo_turbulence_150h5Display.SetScaleArray = ['POINTS', 'vx']
demo_turbulence_150h5Display.ScaleTransferFunction = 'PiecewiseFunction'
demo_turbulence_150h5Display.OpacityArray = ['POINTS', 'vx']
demo_turbulence_150h5Display.OpacityTransferFunction = 'PiecewiseFunction'
demo_turbulence_150h5Display.DataAxesGrid = 'GridAxesRepresentation'
demo_turbulence_150h5Display.PolarAxes = 'PolarAxesRepresentation'
demo_turbulence_150h5Display.SelectInputVectors = [None, '']
demo_turbulence_150h5Display.WriteLog = ''

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
demo_turbulence_150h5Display.ScaleTransferFunction.Points = [-0.0688323900103569, 0.0, 0.5, 0.0, 0.0663590133190155, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
demo_turbulence_150h5Display.OpacityTransferFunction.Points = [-0.0688323900103569, 0.0, 0.5, 0.0, 0.0663590133190155, 1.0, 0.5, 0.0]

# reset view to fit data
renderView1.ResetCamera(False)

# show color bar/color legend
demo_turbulence_150h5Display.SetScalarBarVisibility(renderView1, True)

# update the view to ensure updated data information
renderView1.Update()

# get opacity transfer function/opacity map for 'vx'
vxPWF = GetOpacityTransferFunction('vx')

# get 2D transfer function for 'vx'
vxTF2D = GetTransferFunction2D('vx')

# Properties modified on renderView1
renderView1.OrientationAxesVisibility = 0

# Properties modified on renderView1
renderView1.UseColorPaletteForBackground = 0

# Properties modified on renderView1
renderView1.BackgroundColorMode = 'Gradient'

# Properties modified on renderView1
renderView1.Background = [0.8705882352941177, 0.8666666666666667, 0.8549019607843137]

# create a new 'Extract Time Steps'
extractTimeSteps1 = ExtractTimeSteps(registrationName='ExtractTimeSteps1', Input=demo_turbulence_150h5)
extractTimeSteps1.TimeStepIndices = [0]
extractTimeSteps1.TimeStepRange = [0, 0]

# update animation scene based on data timesteps
animationScene1.UpdateAnimationUsingDataTimeSteps()

# show data in view
extractTimeSteps1Display = Show(extractTimeSteps1, renderView1, 'GeometryRepresentation')

# trace defaults for the display properties.
extractTimeSteps1Display.Representation = 'Surface'
extractTimeSteps1Display.ColorArrayName = ['POINTS', 'vx']
extractTimeSteps1Display.LookupTable = vxLUT
extractTimeSteps1Display.SelectTCoordArray = 'None'
extractTimeSteps1Display.SelectNormalArray = 'None'
extractTimeSteps1Display.SelectTangentArray = 'None'
extractTimeSteps1Display.OSPRayScaleArray = 'vx'
extractTimeSteps1Display.OSPRayScaleFunction = 'PiecewiseFunction'
extractTimeSteps1Display.SelectOrientationVectors = 'None'
extractTimeSteps1Display.ScaleFactor = 0.09999994716636358
extractTimeSteps1Display.SelectScaleArray = 'vx'
extractTimeSteps1Display.GlyphType = 'Arrow'
extractTimeSteps1Display.GlyphTableIndexArray = 'vx'
extractTimeSteps1Display.GaussianRadius = 0.004999997358318179
extractTimeSteps1Display.SetScaleArray = ['POINTS', 'vx']
extractTimeSteps1Display.ScaleTransferFunction = 'PiecewiseFunction'
extractTimeSteps1Display.OpacityArray = ['POINTS', 'vx']
extractTimeSteps1Display.OpacityTransferFunction = 'PiecewiseFunction'
extractTimeSteps1Display.DataAxesGrid = 'GridAxesRepresentation'
extractTimeSteps1Display.PolarAxes = 'PolarAxesRepresentation'
extractTimeSteps1Display.SelectInputVectors = [None, '']
extractTimeSteps1Display.WriteLog = ''

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
extractTimeSteps1Display.ScaleTransferFunction.Points = [-0.0688323900103569, 0.0, 0.5, 0.0, 0.0663590133190155, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
extractTimeSteps1Display.OpacityTransferFunction.Points = [-0.0688323900103569, 0.0, 0.5, 0.0, 0.0663590133190155, 1.0, 0.5, 0.0]

# hide data in view
Hide(demo_turbulence_150h5, renderView1)

# show color bar/color legend
extractTimeSteps1Display.SetScalarBarVisibility(renderView1, True)

# update the view to ensure updated data information
renderView1.Update()

# create a new 'Threshold'
threshold1 = Threshold(registrationName='Threshold1', Input=extractTimeSteps1)
threshold1.Scalars = ['POINTS', 'vx']
threshold1.LowerThreshold = -0.0688323900103569
threshold1.UpperThreshold = 0.0663590133190155

# Properties modified on threshold1
threshold1.LowerThreshold = -0.5
threshold1.UpperThreshold = -0.49

# show data in view
threshold1Display = Show(threshold1, renderView1, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
threshold1Display.Representation = 'Surface'
threshold1Display.ColorArrayName = [None, '']
threshold1Display.SelectTCoordArray = 'None'
threshold1Display.SelectNormalArray = 'None'
threshold1Display.SelectTangentArray = 'None'
threshold1Display.OSPRayScaleFunction = 'PiecewiseFunction'
threshold1Display.SelectOrientationVectors = 'None'
threshold1Display.ScaleFactor = -2.0000000000000002e+298
threshold1Display.SelectScaleArray = 'None'
threshold1Display.GlyphType = 'Arrow'
threshold1Display.GlyphTableIndexArray = 'None'
threshold1Display.GaussianRadius = -1e+297
threshold1Display.SetScaleArray = [None, '']
threshold1Display.ScaleTransferFunction = 'PiecewiseFunction'
threshold1Display.OpacityArray = [None, '']
threshold1Display.OpacityTransferFunction = 'PiecewiseFunction'
threshold1Display.DataAxesGrid = 'GridAxesRepresentation'
threshold1Display.PolarAxes = 'PolarAxesRepresentation'
threshold1Display.OpacityArrayName = [None, '']
threshold1Display.SelectInputVectors = [None, '']
threshold1Display.WriteLog = ''

# hide data in view
Hide(extractTimeSteps1, renderView1)

# update the view to ensure updated data information
renderView1.Update()

# Properties modified on threshold1
threshold1.Scalars = ['POINTS', 'z']

# update the view to ensure updated data information
renderView1.Update()

# create a new 'Point Volume Interpolator'
pointVolumeInterpolator1 = PointVolumeInterpolator(registrationName='PointVolumeInterpolator1', Input=threshold1,
    Source='Bounded Volume')
pointVolumeInterpolator1.Kernel = 'VoronoiKernel'
pointVolumeInterpolator1.Locator = 'Static Point Locator'

# init the 'Bounded Volume' selected for 'Source'
pointVolumeInterpolator1.Source.Origin = [-0.4999912922165422, -0.49980609960924527, -0.4999998824265238]
pointVolumeInterpolator1.Source.Scale = [0.9998139293278856, 0.9997060201883242, 0.009999685442683648]

# show data in view
pointVolumeInterpolator1Display = Show(pointVolumeInterpolator1, renderView1, 'UniformGridRepresentation')

# trace defaults for the display properties.
pointVolumeInterpolator1Display.Representation = 'Outline'
pointVolumeInterpolator1Display.ColorArrayName = ['POINTS', '']
pointVolumeInterpolator1Display.SelectTCoordArray = 'None'
pointVolumeInterpolator1Display.SelectNormalArray = 'None'
pointVolumeInterpolator1Display.SelectTangentArray = 'None'
pointVolumeInterpolator1Display.OSPRayScaleArray = 'vx'
pointVolumeInterpolator1Display.OSPRayScaleFunction = 'PiecewiseFunction'
pointVolumeInterpolator1Display.SelectOrientationVectors = 'None'
pointVolumeInterpolator1Display.ScaleFactor = 0.09998139293278857
pointVolumeInterpolator1Display.SelectScaleArray = 'vx'
pointVolumeInterpolator1Display.GlyphType = 'Arrow'
pointVolumeInterpolator1Display.GlyphTableIndexArray = 'vx'
pointVolumeInterpolator1Display.GaussianRadius = 0.004999069646639428
pointVolumeInterpolator1Display.SetScaleArray = ['POINTS', 'vx']
pointVolumeInterpolator1Display.ScaleTransferFunction = 'PiecewiseFunction'
pointVolumeInterpolator1Display.OpacityArray = ['POINTS', 'vx']
pointVolumeInterpolator1Display.OpacityTransferFunction = 'PiecewiseFunction'
pointVolumeInterpolator1Display.DataAxesGrid = 'GridAxesRepresentation'
pointVolumeInterpolator1Display.PolarAxes = 'PolarAxesRepresentation'
pointVolumeInterpolator1Display.ScalarOpacityUnitDistance = 0.014139094808646146
pointVolumeInterpolator1Display.OpacityArrayName = ['POINTS', 'vx']
pointVolumeInterpolator1Display.ColorArray2Name = ['POINTS', 'vx']
pointVolumeInterpolator1Display.IsosurfaceValues = [-0.003703940659761429]
pointVolumeInterpolator1Display.SliceFunction = 'Plane'
pointVolumeInterpolator1Display.Slice = 50
pointVolumeInterpolator1Display.SelectInputVectors = [None, '']
pointVolumeInterpolator1Display.WriteLog = ''

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
pointVolumeInterpolator1Display.ScaleTransferFunction.Points = [-0.058056220412254333, 0.0, 0.5, 0.0, 0.050648339092731476, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
pointVolumeInterpolator1Display.OpacityTransferFunction.Points = [-0.058056220412254333, 0.0, 0.5, 0.0, 0.050648339092731476, 1.0, 0.5, 0.0]

# init the 'Plane' selected for 'SliceFunction'
pointVolumeInterpolator1Display.SliceFunction.Origin = [-8.432755259940583e-05, 4.691048491678451e-05, -0.4949998824265238]

# hide data in view
Hide(threshold1, renderView1)

# update the view to ensure updated data information
renderView1.Update()

# set active source
SetActiveSource(threshold1)

# toggle interactive widget visibility (only when running from the GUI)
HideInteractiveWidgets(proxy=pointVolumeInterpolator1.Source)

# set active source
SetActiveSource(pointVolumeInterpolator1)

# toggle interactive widget visibility (only when running from the GUI)
ShowInteractiveWidgets(proxy=pointVolumeInterpolator1Display.SliceFunction)

# toggle interactive widget visibility (only when running from the GUI)
ShowInteractiveWidgets(proxy=pointVolumeInterpolator1Display)

# toggle interactive widget visibility (only when running from the GUI)
HideInteractiveWidgets(proxy=pointVolumeInterpolator1Display.SliceFunction)

# toggle interactive widget visibility (only when running from the GUI)
HideInteractiveWidgets(proxy=pointVolumeInterpolator1Display)

# toggle interactive widget visibility (only when running from the GUI)
ShowInteractiveWidgets(proxy=pointVolumeInterpolator1.Source)

# change representation type
pointVolumeInterpolator1Display.SetRepresentationType('Surface')

# set active source
SetActiveSource(threshold1)

# toggle interactive widget visibility (only when running from the GUI)
HideInteractiveWidgets(proxy=pointVolumeInterpolator1.Source)

# show data in view
threshold1Display = Show(threshold1, renderView1, 'UnstructuredGridRepresentation')

# hide data in view
Hide(pointVolumeInterpolator1, renderView1)

# set active source
SetActiveSource(pointVolumeInterpolator1)

# toggle interactive widget visibility (only when running from the GUI)
ShowInteractiveWidgets(proxy=pointVolumeInterpolator1Display.SliceFunction)

# toggle interactive widget visibility (only when running from the GUI)
ShowInteractiveWidgets(proxy=pointVolumeInterpolator1Display)

# toggle interactive widget visibility (only when running from the GUI)
HideInteractiveWidgets(proxy=pointVolumeInterpolator1Display.SliceFunction)

# toggle interactive widget visibility (only when running from the GUI)
HideInteractiveWidgets(proxy=pointVolumeInterpolator1Display)

# toggle interactive widget visibility (only when running from the GUI)
ShowInteractiveWidgets(proxy=pointVolumeInterpolator1.Source)

# show data in view
pointVolumeInterpolator1Display = Show(pointVolumeInterpolator1, renderView1, 'UniformGridRepresentation')

# hide data in view
Hide(threshold1, renderView1)

# set active source
SetActiveSource(pointVolumeInterpolator1)

# set scalar coloring
ColorBy(pointVolumeInterpolator1Display, ('POINTS', 'vx'))

# rescale color and/or opacity maps used to include current data range
pointVolumeInterpolator1Display.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
pointVolumeInterpolator1Display.SetScalarBarVisibility(renderView1, True)

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
vxLUT.ApplyPreset('bone_Matlab', True)

# get color legend/bar for vxLUT in view renderView1
vxLUTColorBar = GetScalarBar(vxLUT, renderView1)

# Properties modified on vxLUTColorBar
vxLUTColorBar.TitleFontSize = 12
vxLUTColorBar.LabelFontSize = 9
vxLUTColorBar.ScalarBarThickness = 10
vxLUTColorBar.ScalarBarLength = 0.2

# change scalar bar placement
vxLUTColorBar.WindowLocation = 'Any Location'
vxLUTColorBar.Position = [0.7995006887052342, 0.7907253269916765]
vxLUTColorBar.ScalarBarLength = 0.20000000000000018

# change scalar bar placement
vxLUTColorBar.Position = [0.7340244982290437, 0.7895362663495838]

# get animation track
threshold1LowerThresholdTrack = GetAnimationTrack('LowerThreshold', index=0, proxy=threshold1)

# create a new key frame
keyFrame8482 = CompositeKeyFrame()
keyFrame8482.KeyValues = [-0.4999998824265238]

# create a new key frame
keyFrame8483 = CompositeKeyFrame()
keyFrame8483.KeyTime = 1.0
keyFrame8483.KeyValues = [0.4999943182082335]

# initialize the animation track
threshold1LowerThresholdTrack.KeyFrames = [keyFrame8482, keyFrame8483]

# Properties modified on keyFrame8482
keyFrame8482.KeyValues = [-0.5]

# Properties modified on keyFrame8483
keyFrame8483.KeyValues = [0.49]

# get animation track
threshold1UpperThresholdTrack = GetAnimationTrack('UpperThreshold', index=0, proxy=threshold1)

# create a new key frame
keyFrame8488 = CompositeKeyFrame()
keyFrame8488.KeyValues = [-0.4999998824265238]

# create a new key frame
keyFrame8489 = CompositeKeyFrame()
keyFrame8489.KeyTime = 1.0
keyFrame8489.KeyValues = [0.4999943182082335]

# initialize the animation track
threshold1UpperThresholdTrack.KeyFrames = [keyFrame8488, keyFrame8489]

# Properties modified on keyFrame8488
keyFrame8488.KeyValues = [-0.49]

# Properties modified on keyFrame8489
keyFrame8489.KeyValues = [0.5]

# get animation track
pointVolumeInterpolator1SourceOriginTrack = GetAnimationTrack('Origin', index=2, proxy=pointVolumeInterpolator1.Source)

# create a new key frame
keyFrame8491 = CompositeKeyFrame()
keyFrame8491.KeyValues = [-0.4999912922165422]

# create a new key frame
keyFrame8492 = CompositeKeyFrame()
keyFrame8492.KeyTime = 1.0
keyFrame8492.KeyValues = [0.49982263711134334]

# initialize the animation track
pointVolumeInterpolator1SourceOriginTrack.KeyFrames = [keyFrame8491, keyFrame8492]

# Properties modified on keyFrame8491
keyFrame8491.KeyValues = [-0.5]

# Properties modified on keyFrame8492
keyFrame8492.KeyValues = [0.49]

# get camera animation track for the view
cameraAnimationCue1 = GetCameraTrack(view=renderView1)

# create a new key frame
keyFrame8494 = CameraKeyFrame()
keyFrame8494.Position = [1.6410553740908718e-07, 6.225440918883329e-07, 1.2900465689940241]
keyFrame8494.FocalPoint = [1.6410553740908718e-07, 6.225440918883329e-07, -2.7821091451718516e-06]
keyFrame8494.ParallelScale = 0.866022961213221
keyFrame8494.PositionPathPoints = [-8.432755259940583e-05, 4.691048491678451e-05, 2.00453494089319, 1.4691053791862005, 4.691048491678451e-05, 1.527166267671106, 2.377114553872269, 4.691048491678451e-05, 0.27739885601124914, 2.3771145538722687, 4.691048491678451e-05, -1.2673986208642967, 1.4691053791862, 4.691048491678451e-05, -2.5171660325241527, -8.432755259918379e-05, 4.691048491678451e-05, -2.994534705746236, -1.4692740342913981, 4.691048491678451e-05, -2.5171660325241523, -2.377283208977466, 4.691048491678451e-05, -1.2673986208642964, -2.377283208977466, 4.691048491678451e-05, 0.2773988560112478, -1.4692740342913981, 4.691048491678451e-05, 1.5271662676711029]
keyFrame8494.FocalPathPoints = [-8.432755259940583e-05, 4.691048491678451e-05, -0.4949998824265238]
keyFrame8494.ClosedPositionPath = 1

# create a new key frame
keyFrame8495 = CameraKeyFrame()
keyFrame8495.KeyTime = 1.0
keyFrame8495.Position = [1.6410553740908718e-07, 6.225440918883329e-07, 1.2900465689940241]
keyFrame8495.FocalPoint = [1.6410553740908718e-07, 6.225440918883329e-07, -2.7821091451718516e-06]
keyFrame8495.ParallelScale = 0.866022961213221

# initialize the animation track
cameraAnimationCue1.Mode = 'Follow-data'
cameraAnimationCue1.KeyFrames = [keyFrame8494, keyFrame8495]
cameraAnimationCue1.DataSource = pointVolumeInterpolator1


# Properties modified on animationScene1
animationScene1.EndTime = 1000.0

# Properties modified on animationScene1
animationScene1.NumberOfFrames = 1000


#================================================================
# addendum: following script captures some of the application
# state to faithfully reproduce the visualization during playback
#================================================================

# get layout
layout1 = GetLayout()

#--------------------------------
# saving layout sizes for layouts

# layout/tab size in pixels
layout1.SetSize(1848, 841)

#-----------------------------------
# saving camera placements for views

# current camera placement for renderView1
renderView1.CameraPosition = [-8.432755259940583e-05, 4.691048491678451e-05, 1.3937612549501512]
renderView1.CameraFocalPoint = [-8.432755259940583e-05, 4.691048491678451e-05, -0.495]
renderView1.CameraParallelScale = 0.866022961213221


# get active view
renderView1 = GetActiveViewOrCreate('RenderView')

# create extractor
pNG1 = CreateExtractor('PNG', renderView1, registrationName='PNG1')
# trace defaults for the extractor.
pNG1.Trigger = 'TimeStep'

# init the 'PNG' selected for 'Writer'
pNG1.Writer.FileName = 'RenderView1_{timestep:06d}{camera}.png'
pNG1.Writer.ImageResolution = [1848, 841]
pNG1.Writer.Format = 'PNG'

#--------------------------------------------
# uncomment the following to render all views
# RenderAllViews()
# alternatively, if you want to write images, you can use SaveScreenshot(...).

if __name__ == '__main__':
    # generate extracts
    SaveExtracts(ExtractsOutputDirectory='/scratch/snx3000/yzhu/vis/extracts')
    # SaveAnimation('/scratch/snx3000/yzhu/vis/test.avi', renderView1, ImageResolution=[1848, 841],
    # FrameWindow=[0, 30], FrameRate=30)
# state file generated using paraview version 5.11.1
import paraview
paraview.compatibility.major = 5
paraview.compatibility.minor = 11

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# ----------------------------------------------------------------
# setup views used in the visualization
# ----------------------------------------------------------------

# Create a new 'Render View'
renderView1 = CreateView('RenderView')
renderView1.ViewSize = [1912, 1864]
renderView1.InteractionMode = '2D'
renderView1.AxesGrid = 'GridAxes3DActor'
renderView1.CenterOfRotation = [-129.37203216552734, -182.3526840209961, -0.0002988874912261963]
renderView1.StereoType = 'Crystal Eyes'
renderView1.CameraPosition = [-129.37203216552734, -182.3526840209961, 3.3128419090228736]
renderView1.CameraFocalPoint = [-129.37203216552734, -182.3526840209961, -0.0002988874912261963]
renderView1.CameraFocalDisk = 1.0
renderView1.CameraParallelScale = 0.8575039372439843

# Create a new 'SpreadSheet View'
spreadSheetView1 = CreateView('SpreadSheetView')
spreadSheetView1.ColumnToSort = ''
spreadSheetView1.BlockSize = 1024

SetActiveView(None)

# ----------------------------------------------------------------
# setup view layouts
# ----------------------------------------------------------------

# create new layout object 'Layout #1'
layout1 = CreateLayout(name='Layout #1')
layout1.SplitHorizontal(0, 0.586081)
layout1.AssignView(1, renderView1)
layout1.AssignView(2, spreadSheetView1)
layout1.SetSize(2313, 1864)

# ----------------------------------------------------------------
# restore active view
SetActiveView(renderView1)
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# setup the data processing pipelines
# ----------------------------------------------------------------

# create a new 'H5PartReader'
tde_snapshot00000h5 = H5PartReader(registrationName='tde_snapshot00000.h5', FileName='/home/appcell/tde_snapshot00000.h5')
tde_snapshot00000h5.Xarray = 'x'
tde_snapshot00000h5.Yarray = 'y'
tde_snapshot00000h5.Zarray = 'z'
tde_snapshot00000h5.PointArrays = ['rho', 'x', 'y', 'z']

# create a new 'Point Volume Interpolator'
pointVolumeInterpolator1 = PointVolumeInterpolator(registrationName='PointVolumeInterpolator1', Input=tde_snapshot00000h5,
    Source='Bounded Volume')
pointVolumeInterpolator1.Kernel = 'VoronoiKernel'
pointVolumeInterpolator1.Locator = 'Static Point Locator'

# init the 'Bounded Volume' selected for 'Source'
pointVolumeInterpolator1.Source.Origin = [-129.86793518066406, -182.8478240966797, -0.4944947063922882]
pointVolumeInterpolator1.Source.Scale = [0.9918060302734375, 0.9902801513671875, 0.988391637802124]

# ----------------------------------------------------------------
# setup the visualization in view 'renderView1'
# ----------------------------------------------------------------

# show data from tde_snapshot00000h5
tde_snapshot00000h5Display = Show(tde_snapshot00000h5, renderView1, 'GeometryRepresentation')

# get 2D transfer function for 'rho'
rhoTF2D = GetTransferFunction2D('rho')
rhoTF2D.ScalarRangeInitialized = 1
rhoTF2D.Range = [1.3986570124302489e-08, 1.3124645192874596e-05, 0.0, 1.0]

# get color transfer function/color map for 'rho'
rhoLUT = GetColorTransferFunction('rho')
rhoLUT.TransferFunction2D = rhoTF2D
rhoLUT.RGBPoints = [0.0, 0.231373, 0.298039, 0.752941, 8.481556506012566e-06, 0.865003, 0.865003, 0.865003, 1.6963113012025133e-05, 0.705882, 0.0156863, 0.14902]
rhoLUT.ScalarRangeInitialized = 1.0

# trace defaults for the display properties.
tde_snapshot00000h5Display.Representation = 'Surface'
tde_snapshot00000h5Display.ColorArrayName = ['POINTS', 'rho']
tde_snapshot00000h5Display.LookupTable = rhoLUT
tde_snapshot00000h5Display.SelectTCoordArray = 'None'
tde_snapshot00000h5Display.SelectNormalArray = 'None'
tde_snapshot00000h5Display.SelectTangentArray = 'None'
tde_snapshot00000h5Display.OSPRayScaleArray = 'rho'
tde_snapshot00000h5Display.OSPRayScaleFunction = 'PiecewiseFunction'
tde_snapshot00000h5Display.SelectOrientationVectors = 'None'
tde_snapshot00000h5Display.ScaleFactor = 0.08983001708984376
tde_snapshot00000h5Display.SelectScaleArray = 'rho'
tde_snapshot00000h5Display.GlyphType = 'Arrow'
tde_snapshot00000h5Display.GlyphTableIndexArray = 'rho'
tde_snapshot00000h5Display.GaussianRadius = 0.004491500854492188
tde_snapshot00000h5Display.SetScaleArray = ['POINTS', 'rho']
tde_snapshot00000h5Display.ScaleTransferFunction = 'PiecewiseFunction'
tde_snapshot00000h5Display.OpacityArray = ['POINTS', 'rho']
tde_snapshot00000h5Display.OpacityTransferFunction = 'PiecewiseFunction'
tde_snapshot00000h5Display.DataAxesGrid = 'GridAxesRepresentation'
tde_snapshot00000h5Display.PolarAxes = 'PolarAxesRepresentation'
tde_snapshot00000h5Display.SelectInputVectors = [None, '']
tde_snapshot00000h5Display.WriteLog = ''

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
tde_snapshot00000h5Display.ScaleTransferFunction.Points = [9.366737430127614e-08, 0.0, 0.5, 0.0, 1.6963113012025133e-05, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
tde_snapshot00000h5Display.OpacityTransferFunction.Points = [9.366737430127614e-08, 0.0, 0.5, 0.0, 1.6963113012025133e-05, 1.0, 0.5, 0.0]

# show data from pointVolumeInterpolator1
pointVolumeInterpolator1Display = Show(pointVolumeInterpolator1, renderView1, 'UniformGridRepresentation')

# get opacity transfer function/opacity map for 'rho'
rhoPWF = GetOpacityTransferFunction('rho')
rhoPWF.Points = [0.0, 0.0, 0.5, 0.0, 1.6963113012025133e-05, 1.0, 0.5, 0.0]
rhoPWF.ScalarRangeInitialized = 1

# trace defaults for the display properties.
pointVolumeInterpolator1Display.Representation = 'Outline'
pointVolumeInterpolator1Display.ColorArrayName = ['POINTS', 'rho']
pointVolumeInterpolator1Display.LookupTable = rhoLUT
pointVolumeInterpolator1Display.SelectTCoordArray = 'None'
pointVolumeInterpolator1Display.SelectNormalArray = 'None'
pointVolumeInterpolator1Display.SelectTangentArray = 'None'
pointVolumeInterpolator1Display.OSPRayScaleArray = 'rho'
pointVolumeInterpolator1Display.OSPRayScaleFunction = 'PiecewiseFunction'
pointVolumeInterpolator1Display.SelectOrientationVectors = 'None'
pointVolumeInterpolator1Display.ScaleFactor = 0.09918060302734376
pointVolumeInterpolator1Display.SelectScaleArray = 'rho'
pointVolumeInterpolator1Display.GlyphType = 'Arrow'
pointVolumeInterpolator1Display.GlyphTableIndexArray = 'rho'
pointVolumeInterpolator1Display.GaussianRadius = 0.004959030151367188
pointVolumeInterpolator1Display.SetScaleArray = ['POINTS', 'rho']
pointVolumeInterpolator1Display.ScaleTransferFunction = 'PiecewiseFunction'
pointVolumeInterpolator1Display.OpacityArray = ['POINTS', 'rho']
pointVolumeInterpolator1Display.OpacityTransferFunction = 'PiecewiseFunction'
pointVolumeInterpolator1Display.DataAxesGrid = 'GridAxesRepresentation'
pointVolumeInterpolator1Display.PolarAxes = 'PolarAxesRepresentation'
pointVolumeInterpolator1Display.ScalarOpacityUnitDistance = 0.017150078744879685
pointVolumeInterpolator1Display.ScalarOpacityFunction = rhoPWF
pointVolumeInterpolator1Display.TransferFunction2D = rhoTF2D
pointVolumeInterpolator1Display.OpacityArrayName = ['POINTS', 'rho']
pointVolumeInterpolator1Display.ColorArray2Name = ['POINTS', 'rho']
pointVolumeInterpolator1Display.IsosurfaceValues = [6.569315881499449e-06]
pointVolumeInterpolator1Display.SliceFunction = 'Plane'
pointVolumeInterpolator1Display.Slice = 50
pointVolumeInterpolator1Display.SelectInputVectors = [None, '']
pointVolumeInterpolator1Display.WriteLog = ''

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
pointVolumeInterpolator1Display.ScaleTransferFunction.Points = [1.3986570124302489e-08, 0.0, 0.5, 0.0, 1.3124645192874596e-05, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
pointVolumeInterpolator1Display.OpacityTransferFunction.Points = [1.3986570124302489e-08, 0.0, 0.5, 0.0, 1.3124645192874596e-05, 1.0, 0.5, 0.0]

# init the 'Plane' selected for 'SliceFunction'
pointVolumeInterpolator1Display.SliceFunction.Origin = [-129.37203216552734, -182.3526840209961, -0.0002988874912261963]

# setup the color legend parameters for each legend in this view

# get color legend/bar for rhoLUT in view renderView1
rhoLUTColorBar = GetScalarBar(rhoLUT, renderView1)
rhoLUTColorBar.Title = 'rho'
rhoLUTColorBar.ComponentTitle = ''

# set color bar visibility
rhoLUTColorBar.Visibility = 1

# show color legend
tde_snapshot00000h5Display.SetScalarBarVisibility(renderView1, True)

# hide data in view
Hide(tde_snapshot00000h5, renderView1)

# show color legend
pointVolumeInterpolator1Display.SetScalarBarVisibility(renderView1, True)

# ----------------------------------------------------------------
# setup color maps and opacity mapes used in the visualization
# note: the Get..() functions create a new object, if needed
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# restore active source
SetActiveSource(pointVolumeInterpolator1)
# ----------------------------------------------------------------


if __name__ == '__main__':
    # generate extracts
    SaveExtracts(ExtractsOutputDirectory='extracts')
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

# Create a new 'Light'
light1 = CreateLight()
light1.Coords = 'Camera'
light1.Intensity = 1.5
light1.Type = 'Positional'
light1.DiffuseColor = [0.6, 0.7568627450980392, 0.9450980392156862]
light1.ConeAngle = 54.9

# a texture
m3_Photoreal_equirectangularjpg_very_very_dark_deep_1616583075_11824881 = CreateTexture('/home/appcell/Downloads/M3_Photoreal_equirectangular-jpg_very_very_dark_deep_1616583075_11824881.jpg')

# create light
# Create a new 'Render View'
renderView1 = CreateView('RenderView')
renderView1.ViewSize = [2427, 613]
renderView1.AxesGrid = 'GridAxes3DActor'
renderView1.OrientationAxesVisibility = 0
renderView1.CenterOfRotation = [-27.789823388549806, -66.9249013885498, 0.0]
renderView1.KeyLightIntensity = 1.0
renderView1.FillLightWarmth = 0.9999999999999999
renderView1.FillLightKFRatio = 7.5
renderView1.BackLightKBRatio = 4.0
renderView1.HeadLightKHRatio = 5.0
renderView1.StereoType = 'Crystal Eyes'
renderView1.CameraPosition = [-126.75824863867895, -119.33424257018956, -9.664040711179856]
renderView1.CameraFocalPoint = [30.09833306556709, -23.96501156459235, 7.390934886158959]
renderView1.CameraViewUp = [-0.10892399116992502, 0.0013836901516328173, 0.9940491182779543]
renderView1.CameraFocalDisk = 1.0
renderView1.CameraParallelScale = 225.768
renderView1.UseColorPaletteForBackground = 0
renderView1.BackgroundColorMode = 'Skybox'
renderView1.BackgroundTexture = m3_Photoreal_equirectangularjpg_very_very_dark_deep_1616583075_11824881
renderView1.Background = [0.05888456549935149, 0.08593881132219425, 0.17666895551995118]
renderView1.UseEnvironmentLighting = 1
renderView1.AdditionalLights = light1

SetActiveView(None)

# ----------------------------------------------------------------
# setup view layouts
# ----------------------------------------------------------------

# create new layout object 'Layout #1'
layout1 = CreateLayout(name='Layout #1')
layout1.AssignView(0, renderView1)
layout1.SetSize(2427, 613)

# ----------------------------------------------------------------
# restore active view
SetActiveView(renderView1)
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# setup the data processing pipelines
# ----------------------------------------------------------------

# create a new 'H5PartReader'
tde_snapshot00000h5 = H5PartReader(registrationName='tde_snapshot00000.h5', FileName='/home/appcell/Visualizations/TidalDisruptionEvent/Datasets/tde_snapshot00000.h5')
tde_snapshot00000h5.Xarray = 'x'
tde_snapshot00000h5.Yarray = 'y'
tde_snapshot00000h5.Zarray = 'z'
tde_snapshot00000h5.PointArrays = ['rho', 'x', 'y', 'z']

# create a new 'Sphere'
sphere1 = Sphere(registrationName='Sphere1')
sphere1.Radius = 25.0
sphere1.ThetaResolution = 32
sphere1.PhiResolution = 32

# create a new 'Point Volume Interpolator'
pointVolumeInterpolator1 = PointVolumeInterpolator(registrationName='PointVolumeInterpolator1', Input=tde_snapshot00000h5,
    Source='Bounded Volume')
pointVolumeInterpolator1.Kernel = 'VoronoiKernel'
pointVolumeInterpolator1.Locator = 'Static Point Locator'

# init the 'Bounded Volume' selected for 'Source'
pointVolumeInterpolator1.Source.Origin = [-70.560394, -148.83055, -0.47299883]
pointVolumeInterpolator1.Source.Scale = [0.9639206, 1.0030518, 0.9513724]

# create a new 'Threshold'
threshold1 = Threshold(registrationName='Threshold1', Input=tde_snapshot00000h5)
threshold1.Scalars = ['POINTS', 'rho']
threshold1.LowerThreshold = 2e-06
threshold1.UpperThreshold = 1.6963113012025133e-05
threshold1.ThresholdMethod = 'Below Lower Threshold'

# ----------------------------------------------------------------
# setup the visualization in view 'renderView1'
# ----------------------------------------------------------------

# show data from tde_snapshot00000h5
tde_snapshot00000h5Display = Show(tde_snapshot00000h5, renderView1, 'GeometryRepresentation')

# get 2D transfer function for 'rho'
rhoTF2D = GetTransferFunction2D('rho')
rhoTF2D.ScalarRangeInitialized = 1
rhoTF2D.Range = [9.366737430127614e-08, 1.5907302440609783e-05, 0.0, 1.0]

# get color transfer function/color map for 'rho'
rhoLUT = GetColorTransferFunction('rho')
rhoLUT.AutomaticRescaleRangeMode = 'Clamp and update every timestep'
rhoLUT.TransferFunction2D = rhoTF2D
rhoLUT.RGBPoints = [1.672478333603067e-08, 0.001462, 0.000466, 0.013866, 6.598302682533586e-08, 0.002267, 0.00127, 0.01857, 1.1522871084409307e-07, 0.003299, 0.002249, 0.024239, 1.6448695433339855e-07, 0.004547, 0.003392, 0.030909, 2.137326383521554e-07, 0.006006, 0.004692, 0.038558, 2.6299088184145997e-07, 0.007676, 0.006136, 0.046836, 3.122365658602181e-07, 0.009561, 0.007713, 0.055143, 3.6149480934952313e-07, 0.011663, 0.009417, 0.06346, 4.10753052838828e-07, 0.013995, 0.011225, 0.071862, 4.599987368575852e-07, 0.016561, 0.013136, 0.080282, 5.092569803468894e-07, 0.019373, 0.015133, 0.088767, 5.585026643656484e-07, 0.022447, 0.017199, 0.097327, 6.077609078549531e-07, 0.025793, 0.019331, 0.10593, 6.570065918737105e-07, 0.029432, 0.021503, 0.114621, 7.062648353630156e-07, 0.033385, 0.023702, 0.123397, 7.555230788523195e-07, 0.037668, 0.025921, 0.132232, 8.04768762871077e-07, 0.042253, 0.028139, 0.141141, 8.54027006360383e-07, 0.046915, 0.030324, 0.150164, 9.032726903791399e-07, 0.051644, 0.032474, 0.159254, 9.525309338684462e-07, 0.056449, 0.034569, 0.168414, 1.001776617887203e-06, 0.06134, 0.03659, 0.177642, 1.0510348613765071e-06, 0.066331, 0.038504, 0.186962, 1.1002931048658117e-06, 0.071429, 0.040294, 0.196354, 1.1495387888845688e-06, 0.076637, 0.041905, 0.205799, 1.1987970323738756e-06, 0.081962, 0.043328, 0.215289, 1.2480427163926338e-06, 0.087411, 0.044556, 0.224813, 1.297300959881939e-06, 0.09299, 0.045583, 0.234358, 1.3465466439006934e-06, 0.098702, 0.046402, 0.243904, 1.3958048873900005e-06, 0.104551, 0.047008, 0.25343, 1.445050571408759e-06, 0.110536, 0.047399, 0.262912, 1.4943088148980635e-06, 0.116656, 0.047574, 0.272321, 1.5435670583873677e-06, 0.122908, 0.047536, 0.281624, 1.592812742406124e-06, 0.129285, 0.047293, 0.290788, 1.6420709858954292e-06, 0.135778, 0.046856, 0.299776, 1.691316669914187e-06, 0.142378, 0.046242, 0.308553, 1.7405749134034925e-06, 0.149073, 0.045468, 0.317085, 1.7898205974222487e-06, 0.15585, 0.044559, 0.325338, 1.839078840911556e-06, 0.162689, 0.043554, 0.333277, 1.8883370844008618e-06, 0.169575, 0.042489, 0.340874, 1.9375827684196185e-06, 0.176493, 0.041402, 0.348111, 1.9868410119089235e-06, 0.183429, 0.040329, 0.354971, 2.0360866959276764e-06, 0.190367, 0.039309, 0.361447, 2.0853449394169836e-06, 0.197297, 0.0384, 0.367535, 2.134590623435742e-06, 0.204209, 0.037632, 0.373238, 2.1838488669250495e-06, 0.211095, 0.03703, 0.378563, 2.233107110414356e-06, 0.217949, 0.036615, 0.383522, 2.282352794433107e-06, 0.224763, 0.036405, 0.388129, 2.331611037922413e-06, 0.231538, 0.036405, 0.3924, 2.3808567219411713e-06, 0.238273, 0.036621, 0.396353, 2.4301149654304763e-06, 0.244967, 0.037055, 0.400007, 2.479360649449237e-06, 0.25162, 0.037705, 0.403378, 2.52861889293854e-06, 0.258234, 0.038571, 0.406485, 2.5778771364278473e-06, 0.26481, 0.039647, 0.409345, 2.6271228204466e-06, 0.271347, 0.040922, 0.411976, 2.6763810639359074e-06, 0.27785, 0.042353, 0.414392, 2.7256267479546637e-06, 0.284321, 0.043933, 0.416608, 2.7748849914439704e-06, 0.290763, 0.045644, 0.418637, 2.824130675462724e-06, 0.297178, 0.04747, 0.420491, 2.8733889189520296e-06, 0.303568, 0.049396, 0.422182, 2.922647162441339e-06, 0.309935, 0.051407, 0.423721, 2.9718928464600964e-06, 0.316282, 0.05349, 0.425116, 3.021151089949398e-06, 0.32261, 0.055634, 0.426377, 3.0703967739681564e-06, 0.328921, 0.057827, 0.427511, 3.1196550174574615e-06, 0.335217, 0.06006, 0.428524, 3.168900701476218e-06, 0.3415, 0.062325, 0.429425, 3.2181589449655245e-06, 0.347771, 0.064616, 0.430217, 3.2674171884548295e-06, 0.354032, 0.066925, 0.430906, 3.3166628724735854e-06, 0.360284, 0.069247, 0.431497, 3.365921115962891e-06, 0.366529, 0.071579, 0.431994, 3.415166799981648e-06, 0.372768, 0.073915, 0.4324, 3.4644250434709543e-06, 0.379001, 0.076253, 0.432719, 3.513670727489711e-06, 0.385228, 0.078591, 0.432955, 3.562928970979016e-06, 0.391453, 0.080927, 0.433109, 3.61218721446832e-06, 0.397674, 0.083257, 0.433183, 3.661432898487081e-06, 0.403894, 0.08558, 0.433179, 3.7106911419763802e-06, 0.410113, 0.087896, 0.433098, 3.759936825995137e-06, 0.416331, 0.090203, 0.432943, 3.80919506948444e-06, 0.422549, 0.092501, 0.432714, 3.858440753503206e-06, 0.428768, 0.09479, 0.432412, 3.907698996992508e-06, 0.434987, 0.097069, 0.432039, 3.9569446810112706e-06, 0.441207, 0.099338, 0.431594, 4.006202924500572e-06, 0.447428, 0.101597, 0.43108, 4.0554611679898755e-06, 0.453651, 0.103848, 0.430498, 4.104706852008631e-06, 0.459875, 0.106089, 0.429846, 4.1539650954979364e-06, 0.4661, 0.108322, 0.429125, 4.203210779516695e-06, 0.472328, 0.110547, 0.428334, 4.252469023005994e-06, 0.478558, 0.112764, 0.427475, 4.3017147070247595e-06, 0.484789, 0.114974, 0.426548, 4.350972950514068e-06, 0.491022, 0.117179, 0.425552, 4.400231194003369e-06, 0.497257, 0.119379, 0.424488, 4.449476878022128e-06, 0.503493, 0.121575, 0.423356, 4.498735121511436e-06, 0.50973, 0.123769, 0.422156, 4.547980805530183e-06, 0.515967, 0.12596, 0.420887, 4.597239049019501e-06, 0.522206, 0.12815, 0.419549, 4.646484733038254e-06, 0.528444, 0.130341, 0.418142, 4.695742976527556e-06, 0.534683, 0.132534, 0.416667, 4.745001220016871e-06, 0.54092, 0.134729, 0.415123, 4.794246904035627e-06, 0.547157, 0.136929, 0.413511, 4.843505147524922e-06, 0.553392, 0.139134, 0.411829, 4.892750831543674e-06, 0.559624, 0.141346, 0.410078, 4.942009075032988e-06, 0.565854, 0.143567, 0.408258, 4.991254759051734e-06, 0.572081, 0.145797, 0.406369, 5.04051300254105e-06, 0.578304, 0.148039, 0.404411, 5.089771246030348e-06, 0.584521, 0.150294, 0.402385, 5.139016930049113e-06, 0.590734, 0.152563, 0.40029, 5.188275173538422e-06, 0.59694, 0.154848, 0.398125, 5.237520857557169e-06, 0.603139, 0.157151, 0.395891, 5.286779101046482e-06, 0.60933, 0.159474, 0.393589, 5.336024785065229e-06, 0.615513, 0.161817, 0.391219, 5.385283028554543e-06, 0.621685, 0.164184, 0.388781, 5.43454127204385e-06, 0.627847, 0.166575, 0.386276, 5.48378695606261e-06, 0.633998, 0.168992, 0.383704, 5.53304519955191e-06, 0.640135, 0.171438, 0.381065, 5.5822908835706715e-06, 0.64626, 0.173914, 0.378359, 5.631549127059975e-06, 0.652369, 0.176421, 0.375586, 5.680794811078725e-06, 0.658463, 0.178962, 0.372748, 5.7300530545680285e-06, 0.66454, 0.181539, 0.369846, 5.779311298057342e-06, 0.670599, 0.184153, 0.366879, 5.82855698207609e-06, 0.676638, 0.186807, 0.363849, 5.877815225565388e-06, 0.682656, 0.189501, 0.360757, 5.927060909584162e-06, 0.688653, 0.192239, 0.357603, 5.976319153073456e-06, 0.694627, 0.195021, 0.354388, 6.025564837092213e-06, 0.700576, 0.197851, 0.351113, 6.074823080581515e-06, 0.7065, 0.200728, 0.347777, 6.124081324070833e-06, 0.712396, 0.203656, 0.344383, 6.173327008089586e-06, 0.718264, 0.206636, 0.340931, 6.222585251578892e-06, 0.724103, 0.20967, 0.337424, 6.271830935597656e-06, 0.729909, 0.212759, 0.333861, 6.321089179086954e-06, 0.735683, 0.215906, 0.330245, 6.37033486310571e-06, 0.741423, 0.219112, 0.326576, 6.419593106595018e-06, 0.747127, 0.222378, 0.322856, 6.468838790613773e-06, 0.752794, 0.225706, 0.319085, 6.518097034103079e-06, 0.758422, 0.229097, 0.315266, 6.567355277592383e-06, 0.76401, 0.232554, 0.311399, 6.61660096161114e-06, 0.769556, 0.236077, 0.307485, 6.665859205100446e-06, 0.775059, 0.239667, 0.303526, 6.715104889119203e-06, 0.780517, 0.243327, 0.299523, 6.764363132608511e-06, 0.785929, 0.247056, 0.295477, 6.813608816627265e-06, 0.791293, 0.250856, 0.29139, 6.862867060116571e-06, 0.796607, 0.254728, 0.287264, 6.912125303605878e-06, 0.801871, 0.258674, 0.283099, 6.961370987624633e-06, 0.807082, 0.262692, 0.278898, 7.010629231113937e-06, 0.812239, 0.266786, 0.274661, 7.0598749151326925e-06, 0.817341, 0.270954, 0.27039, 7.109133158622003e-06, 0.822386, 0.275197, 0.266085, 7.158378842640758e-06, 0.827372, 0.279517, 0.26175, 7.207637086130063e-06, 0.832299, 0.283913, 0.257383, 7.256895329619369e-06, 0.837165, 0.288385, 0.252988, 7.306141013638127e-06, 0.841969, 0.292933, 0.248564, 7.355399257127431e-06, 0.846709, 0.297559, 0.244113, 7.404644941146189e-06, 0.851384, 0.30226, 0.239636, 7.4539031846355e-06, 0.855992, 0.307038, 0.235133, 7.5031488686542415e-06, 0.860533, 0.311892, 0.230606, 7.552407112143554e-06, 0.865006, 0.316822, 0.226055, 7.601665355632851e-06, 0.869409, 0.321827, 0.221482, 7.650911039651621e-06, 0.873741, 0.326906, 0.216886, 7.700169283140917e-06, 0.878001, 0.33206, 0.212268, 7.749414967159682e-06, 0.882188, 0.337287, 0.207628, 7.798673210648986e-06, 0.886302, 0.342586, 0.202968, 7.847918894667747e-06, 0.890341, 0.347957, 0.198286, 7.897177138157032e-06, 0.894305, 0.353399, 0.193584, 7.946435381646344e-06, 0.898192, 0.358911, 0.18886, 7.995681065665113e-06, 0.902003, 0.364492, 0.184116, 8.044939309154418e-06, 0.905735, 0.37014, 0.17935, 8.094184993173167e-06, 0.90939, 0.375856, 0.174563, 8.143443236662459e-06, 0.912966, 0.381636, 0.169755, 8.192688920681231e-06, 0.916462, 0.387481, 0.164924, 8.241947164170545e-06, 0.919879, 0.393389, 0.16007, 8.291205407659842e-06, 0.923215, 0.399359, 0.155193, 8.340451091678596e-06, 0.92647, 0.405389, 0.150292, 8.389709335167915e-06, 0.929644, 0.411479, 0.145367, 8.43895501918667e-06, 0.932737, 0.417627, 0.140417, 8.488213262675957e-06, 0.935747, 0.423831, 0.13544, 8.53745894669471e-06, 0.938675, 0.430091, 0.130438, 8.586717190184052e-06, 0.941521, 0.436405, 0.125409, 8.635975433673322e-06, 0.944285, 0.442772, 0.120354, 8.685221117692106e-06, 0.946965, 0.449191, 0.115272, 8.734479361181398e-06, 0.949562, 0.45566, 0.110164, 8.783725045200167e-06, 0.952075, 0.462178, 0.105031, 8.832983288689464e-06, 0.954506, 0.468744, 0.099874, 8.882228972708224e-06, 0.956852, 0.475356, 0.094695, 8.931487216197518e-06, 0.959114, 0.482014, 0.089499, 8.980732900216284e-06, 0.961293, 0.488716, 0.084289, 9.029991143705576e-06, 0.963387, 0.495462, 0.079073, 9.079249387194896e-06, 0.965397, 0.502249, 0.073859, 9.12849507121364e-06, 0.967322, 0.509078, 0.068659, 9.17775331470297e-06, 0.969163, 0.515946, 0.063488, 9.226998998721752e-06, 0.970919, 0.522853, 0.058367, 9.276257242211015e-06, 0.97259, 0.529798, 0.053324, 9.325502926229767e-06, 0.974176, 0.53678, 0.048392, 9.37476116971908e-06, 0.975677, 0.543798, 0.043618, 9.4240194132084e-06, 0.977092, 0.55085, 0.03905, 9.473265097227133e-06, 0.978422, 0.557937, 0.034931, 9.52252334071644e-06, 0.979666, 0.565057, 0.031409, 9.571769024735223e-06, 0.980824, 0.572209, 0.028508, 9.621027268224498e-06, 0.981895, 0.579392, 0.02625, 9.670272952243267e-06, 0.982881, 0.586606, 0.024661, 9.719531195732576e-06, 0.983779, 0.593849, 0.02377, 9.768789439221896e-06, 0.984591, 0.601122, 0.023606, 9.818035123240626e-06, 0.985315, 0.608422, 0.024202, 9.867293366729945e-06, 0.985952, 0.61575, 0.025592, 9.916539050748699e-06, 0.986502, 0.623105, 0.027814, 9.96579729423799e-06, 0.986964, 0.630485, 0.030908, 1.0015042978256758e-05, 0.987337, 0.63789, 0.034916, 1.0064301221746069e-05, 0.987622, 0.64532, 0.039886, 1.0113559465235357e-05, 0.987819, 0.652773, 0.045581, 1.0162805149254123e-05, 0.987926, 0.66025, 0.05175, 1.0212063392743427e-05, 0.987945, 0.667748, 0.058329, 1.0261309076762196e-05, 0.987874, 0.675267, 0.065257, 1.031056732025149e-05, 0.987714, 0.682807, 0.072489, 1.0359813004270267e-05, 0.987464, 0.690366, 0.07999, 1.040907124775957e-05, 0.987124, 0.697944, 0.087731, 1.0458329491248859e-05, 0.986694, 0.70554, 0.095694, 1.05075751752676e-05, 0.986175, 0.713153, 0.103863, 1.0556833418756933e-05, 0.985566, 0.720782, 0.112229, 1.0606079102775685e-05, 0.984865, 0.728427, 0.120785, 1.0655337346264994e-05, 0.984075, 0.736087, 0.129527, 1.0704583030283723e-05, 0.983196, 0.743758, 0.138453, 1.0753841273773055e-05, 0.982228, 0.751442, 0.147565, 1.0803099517262356e-05, 0.981173, 0.759135, 0.156863, 1.0852345201281114e-05, 0.980032, 0.766837, 0.166353, 1.0901603444770418e-05, 0.978806, 0.774545, 0.176037, 1.0950849128789189e-05, 0.977497, 0.782258, 0.185923, 1.1000107372278503e-05, 0.976108, 0.789974, 0.196018, 1.1049353056297248e-05, 0.974638, 0.797692, 0.206332, 1.1098611299786535e-05, 0.973088, 0.805409, 0.216877, 1.114786954327585e-05, 0.971468, 0.813122, 0.227658, 1.1197115227294587e-05, 0.969783, 0.820825, 0.238686, 1.1246373470783928e-05, 0.968041, 0.828515, 0.249972, 1.129561915480264e-05, 0.966243, 0.836191, 0.261534, 1.1344877398291972e-05, 0.964394, 0.843848, 0.273391, 1.139412308231075e-05, 0.962517, 0.851476, 0.285546, 1.1443381325800026e-05, 0.960626, 0.859069, 0.29801, 1.1492627009818795e-05, 0.95872, 0.866624, 0.31082, 1.1541885253308111e-05, 0.956834, 0.874129, 0.323974, 1.1591143496797403e-05, 0.954997, 0.881569, 0.337475, 1.164038918081615e-05, 0.953215, 0.888942, 0.351369, 1.1689647424305465e-05, 0.951546, 0.896226, 0.365627, 1.1738893108324217e-05, 0.950018, 0.903409, 0.380271, 1.1788151351813533e-05, 0.948683, 0.910473, 0.395289, 1.1837397035832294e-05, 0.947594, 0.917399, 0.410665, 1.1886655279321582e-05, 0.946809, 0.924168, 0.426373, 1.193591352281088e-05, 0.946392, 0.930761, 0.442367, 1.1985159206829648e-05, 0.946403, 0.937159, 0.458592, 1.2034417450318967e-05, 0.946903, 0.943348, 0.47497, 1.2083663134337724e-05, 0.947937, 0.949318, 0.491426, 1.2132921377827e-05, 0.949545, 0.955063, 0.50786, 1.218216706184577e-05, 0.95174, 0.960587, 0.524203, 1.2231425305335077e-05, 0.954529, 0.965896, 0.540361, 1.22806835488244e-05, 0.957896, 0.971003, 0.556275, 1.2329929232843141e-05, 0.961812, 0.975924, 0.571925, 1.2379187476332447e-05, 0.966249, 0.980678, 0.587206, 1.242843316035121e-05, 0.971162, 0.985282, 0.602154, 1.2477691403840525e-05, 0.976511, 0.989753, 0.61676, 1.2526937087859282e-05, 0.982257, 0.994109, 0.631017, 1.2576195331348572e-05, 0.988362, 0.998364, 0.644924]
rhoLUT.NanColor = [0.0, 1.0, 0.0]
rhoLUT.ScalarRangeInitialized = 1.0

# trace defaults for the display properties.
tde_snapshot00000h5Display.Representation = 'Surface'
tde_snapshot00000h5Display.ColorArrayName = ['POINTS', 'rho']
tde_snapshot00000h5Display.LookupTable = rhoLUT
tde_snapshot00000h5Display.Opacity = 0.12
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
tde_snapshot00000h5Display.SelectInputVectors = ['POINTS', '']
tde_snapshot00000h5Display.WriteLog = ''

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
tde_snapshot00000h5Display.ScaleTransferFunction.Points = [9.366737430127614e-08, 0.0, 0.5, 0.0, 1.6963113012025133e-05, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
tde_snapshot00000h5Display.OpacityTransferFunction.Points = [9.366737430127614e-08, 0.0, 0.5, 0.0, 1.6963113012025133e-05, 1.0, 0.5, 0.0]

# show data from pointVolumeInterpolator1
pointVolumeInterpolator1Display = Show(pointVolumeInterpolator1, renderView1, 'UniformGridRepresentation')

# get opacity transfer function/opacity map for 'rho'
rhoPWF = GetOpacityTransferFunction('rho')
rhoPWF.Points = [1.672478333603067e-08, 0.0, 0.5, 0.0, 7.261172914711541e-07, 0.0, 0.5, 0.0, 9.195879329126911e-07, 0.3913043439388275, 0.5, 0.0, 1.1453037132014288e-06, 0.0, 0.5, 0.0, 4.998594249627785e-06, 0.005434782709926367, 0.5, 0.0, 8.996988071885426e-06, 0.1195652186870575, 0.5, 0.0, 1.0907828079333095e-05, 0.2554347813129425, 0.5, 0.0, 1.2069011654233724e-05, 1.0, 0.5, 0.0, 1.2576195331348572e-05, 1.0, 0.5, 0.0]
rhoPWF.ScalarRangeInitialized = 1

# trace defaults for the display properties.
pointVolumeInterpolator1Display.Representation = 'Volume'
pointVolumeInterpolator1Display.ColorArrayName = ['POINTS', 'rho']
pointVolumeInterpolator1Display.LookupTable = rhoLUT
pointVolumeInterpolator1Display.SelectTCoordArray = 'None'
pointVolumeInterpolator1Display.SelectNormalArray = 'None'
pointVolumeInterpolator1Display.SelectTangentArray = 'None'
pointVolumeInterpolator1Display.OSPRayScaleArray = 'rho'
pointVolumeInterpolator1Display.OSPRayScaleFunction = 'PiecewiseFunction'
pointVolumeInterpolator1Display.SelectOrientationVectors = 'None'
pointVolumeInterpolator1Display.ScaleFactor = 0.08983001708984376
pointVolumeInterpolator1Display.SelectScaleArray = 'rho'
pointVolumeInterpolator1Display.GlyphType = 'Arrow'
pointVolumeInterpolator1Display.GlyphTableIndexArray = 'rho'
pointVolumeInterpolator1Display.GaussianRadius = 0.004491500854492188
pointVolumeInterpolator1Display.SetScaleArray = ['POINTS', 'rho']
pointVolumeInterpolator1Display.ScaleTransferFunction = 'PiecewiseFunction'
pointVolumeInterpolator1Display.OpacityArray = ['POINTS', 'rho']
pointVolumeInterpolator1Display.OpacityTransferFunction = 'PiecewiseFunction'
pointVolumeInterpolator1Display.DataAxesGrid = 'GridAxesRepresentation'
pointVolumeInterpolator1Display.PolarAxes = 'PolarAxesRepresentation'
pointVolumeInterpolator1Display.ScalarOpacityUnitDistance = 0.015529270373427327
pointVolumeInterpolator1Display.ScalarOpacityFunction = rhoPWF
pointVolumeInterpolator1Display.TransferFunction2D = rhoTF2D
pointVolumeInterpolator1Display.OpacityArrayName = ['POINTS', 'rho']
pointVolumeInterpolator1Display.ColorArray2Name = ['POINTS', 'rho']
pointVolumeInterpolator1Display.Shade = 1
pointVolumeInterpolator1Display.GlobalIlluminationReach = 0.18
pointVolumeInterpolator1Display.VolumetricScatteringBlending = 0.28
pointVolumeInterpolator1Display.IsosurfaceValues = [8.00048490745553e-06]
pointVolumeInterpolator1Display.SliceFunction = 'Plane'
pointVolumeInterpolator1Display.Slice = 50
pointVolumeInterpolator1Display.SelectInputVectors = ['POINTS', '']
pointVolumeInterpolator1Display.WriteLog = ''

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
pointVolumeInterpolator1Display.ScaleTransferFunction.Points = [9.366737430127614e-08, 0.0, 0.5, 0.0, 1.5907302440609783e-05, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
pointVolumeInterpolator1Display.OpacityTransferFunction.Points = [9.366737430127614e-08, 0.0, 0.5, 0.0, 1.5907302440609783e-05, 1.0, 0.5, 0.0]

# init the 'Plane' selected for 'SliceFunction'
pointVolumeInterpolator1Display.SliceFunction.Origin = [-141.15701293945312, -188.40384674072266, 0.000876054167747442]

# show data from threshold1
threshold1Display = Show(threshold1, renderView1, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
threshold1Display.Representation = 'Points'
threshold1Display.AmbientColor = [0.18823529411764706, 0.2627450980392157, 0.5254901960784314]
threshold1Display.ColorArrayName = ['POINTS', '']
threshold1Display.DiffuseColor = [0.18823529411764706, 0.2627450980392157, 0.5254901960784314]
threshold1Display.Opacity = 0.01
threshold1Display.SelectTCoordArray = 'None'
threshold1Display.SelectNormalArray = 'None'
threshold1Display.SelectTangentArray = 'None'
threshold1Display.OSPRayScaleArray = 'rho'
threshold1Display.OSPRayScaleFunction = 'PiecewiseFunction'
threshold1Display.SelectOrientationVectors = 'None'
threshold1Display.ScaleFactor = 0.08983001708984376
threshold1Display.SelectScaleArray = 'rho'
threshold1Display.GlyphType = 'Arrow'
threshold1Display.GlyphTableIndexArray = 'rho'
threshold1Display.GaussianRadius = 0.004491500854492188
threshold1Display.SetScaleArray = ['POINTS', 'rho']
threshold1Display.ScaleTransferFunction = 'PiecewiseFunction'
threshold1Display.OpacityArray = ['POINTS', 'rho']
threshold1Display.OpacityTransferFunction = 'PiecewiseFunction'
threshold1Display.DataAxesGrid = 'GridAxesRepresentation'
threshold1Display.PolarAxes = 'PolarAxesRepresentation'
threshold1Display.ScalarOpacityUnitDistance = 0.025306868557695913
threshold1Display.OpacityArrayName = ['POINTS', 'rho']
threshold1Display.SelectInputVectors = ['POINTS', '']
threshold1Display.WriteLog = ''

# set separate color map
threshold1Display.UseSeparateColorMap = True

# show data from sphere1
sphere1Display = Show(sphere1, renderView1, 'GeometryRepresentation')

# get separate 2D transfer function for 'Normals'
separate_sphere1Display_NormalsTF2D = GetTransferFunction2D('Normals', sphere1Display, separate=True)
separate_sphere1Display_NormalsTF2D.ScalarRangeInitialized = 1
separate_sphere1Display_NormalsTF2D.Range = [-1.0127336778678, 1.0, -1.0127336778678, 1.0]

# get separate color transfer function/color map for 'Normals'
separate_sphere1Display_NormalsLUT = GetColorTransferFunction('Normals', sphere1Display, separate=True)
separate_sphere1Display_NormalsLUT.TransferFunction2D = separate_sphere1Display_NormalsTF2D
separate_sphere1Display_NormalsLUT.RGBPoints = [-1.0127336778678, 0.0, 0.03, 0.1, 0.028513686099509883, 0.2, 0.2, 0.2, 1.0, 0.0, 0.03, 0.1]
separate_sphere1Display_NormalsLUT.ColorSpace = 'RGB'
separate_sphere1Display_NormalsLUT.NanColor = [1.0, 0.0, 0.0]
separate_sphere1Display_NormalsLUT.ScalarRangeInitialized = 1.0
separate_sphere1Display_NormalsLUT.VectorComponent = 2
separate_sphere1Display_NormalsLUT.VectorMode = 'Component'

# trace defaults for the display properties.
sphere1Display.Representation = 'Surface'
sphere1Display.AmbientColor = [0.0, 0.0, 0.0]
sphere1Display.ColorArrayName = ['POINTS', 'Normals']
sphere1Display.DiffuseColor = [0.0, 0.0, 0.0]
sphere1Display.LookupTable = separate_sphere1Display_NormalsLUT
sphere1Display.Opacity = 0.33
sphere1Display.Specular = 0.21
sphere1Display.SelectTCoordArray = 'None'
sphere1Display.SelectNormalArray = 'Normals'
sphere1Display.SelectTangentArray = 'None'
sphere1Display.OSPRayScaleArray = 'Normals'
sphere1Display.OSPRayScaleFunction = 'PiecewiseFunction'
sphere1Display.SelectOrientationVectors = 'None'
sphere1Display.ScaleFactor = 3.0
sphere1Display.SelectScaleArray = 'None'
sphere1Display.GlyphType = 'Arrow'
sphere1Display.GlyphTableIndexArray = 'None'
sphere1Display.GaussianRadius = 0.15
sphere1Display.SetScaleArray = ['POINTS', 'Normals']
sphere1Display.ScaleTransferFunction = 'PiecewiseFunction'
sphere1Display.OpacityArray = ['POINTS', 'Normals']
sphere1Display.OpacityTransferFunction = 'PiecewiseFunction'
sphere1Display.DataAxesGrid = 'GridAxesRepresentation'
sphere1Display.PolarAxes = 'PolarAxesRepresentation'
sphere1Display.SelectInputVectors = ['POINTS', 'Normals']
sphere1Display.WriteLog = ''

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
sphere1Display.ScaleTransferFunction.Points = [-0.9987165331840515, 0.0, 0.5, 0.0, 0.9987165331840515, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
sphere1Display.OpacityTransferFunction.Points = [-0.9987165331840515, 0.0, 0.5, 0.0, 0.9987165331840515, 1.0, 0.5, 0.0]

# set separate color map
sphere1Display.UseSeparateColorMap = True

# setup the color legend parameters for each legend in this view

# get color legend/bar for rhoLUT in view renderView1
rhoLUTColorBar = GetScalarBar(rhoLUT, renderView1)
rhoLUTColorBar.Position = [0.9542645241038319, 0.009084027252081756]
rhoLUTColorBar.Title = 'rho'
rhoLUTColorBar.ComponentTitle = ''

# set color bar visibility
rhoLUTColorBar.Visibility = 1

# get separate 2D transfer function for 'rho'
separate_threshold1Display_rhoTF2D = GetTransferFunction2D('rho', threshold1Display, separate=True)

# get separate color transfer function/color map for 'rho'
separate_threshold1Display_rhoLUT = GetColorTransferFunction('rho', threshold1Display, separate=True)
separate_threshold1Display_rhoLUT.TransferFunction2D = separate_threshold1Display_rhoTF2D
separate_threshold1Display_rhoLUT.RGBPoints = [1.5612487302973932e-08, 0.107704, 0.107708, 0.107695, 1.401225533394079e-07, 0.141522, 0.13066, 0.270741, 2.646326193758419e-07, 0.180123, 0.146119, 0.42308, 3.8914367760341494e-07, 0.210161, 0.169674, 0.551795, 5.13653743639849e-07, 0.239701, 0.212939, 0.634969, 6.38163809676283e-07, 0.253916, 0.282947, 0.653641, 7.626738757127169e-07, 0.242791, 0.366933, 0.608521, 8.871839417491509e-07, 0.226302, 0.446776, 0.52693, 1.011694573334534e-06, 0.236237, 0.514689, 0.458798, 1.136205066013158e-06, 0.274641, 0.577589, 0.376069, 1.260715132049592e-06, 0.349625, 0.633993, 0.288131, 1.385225198086026e-06, 0.4437, 0.683677, 0.260497, 1.5097352641224601e-06, 0.536247, 0.731214, 0.285424, 1.634246322350033e-06, 0.628472, 0.777128, 0.349151, 1.758756388386467e-06, 0.718259, 0.819287, 0.496825, 1.8832664544229012e-06, 0.804768, 0.856164, 0.703299, 1.9999947653559502e-06, 0.887571, 0.887591, 0.887548]
separate_threshold1Display_rhoLUT.ColorSpace = 'Lab'
separate_threshold1Display_rhoLUT.NanColor = [1.0, 0.0, 0.0]
separate_threshold1Display_rhoLUT.ScalarRangeInitialized = 1.0

# get color legend/bar for separate_threshold1Display_rhoLUT in view renderView1
separate_threshold1Display_rhoLUTColorBar = GetScalarBar(separate_threshold1Display_rhoLUT, renderView1)
separate_threshold1Display_rhoLUTColorBar.WindowLocation = 'Upper Right Corner'
separate_threshold1Display_rhoLUTColorBar.Title = 'rho'
separate_threshold1Display_rhoLUTColorBar.ComponentTitle = ''

# set color bar visibility
separate_threshold1Display_rhoLUTColorBar.Visibility = 0

# get 2D transfer function for 'Normals'
normalsTF2D = GetTransferFunction2D('Normals')
normalsTF2D.ScalarRangeInitialized = 1
normalsTF2D.Range = [1.000000023489076, 1.0000000290839508, 1.000000023489076, 1.0000000290839508]

# get color transfer function/color map for 'Normals'
normalsLUT = GetColorTransferFunction('Normals')
normalsLUT.TransferFunction2D = normalsTF2D
normalsLUT.RGBPoints = [0.9999999680464912, 0.301961, 0.047059, 0.090196, 0.9999999692567254, 0.396078431372549, 0.0392156862745098, 0.058823529411764705, 0.9999999704669627, 0.49411764705882355, 0.054901960784313725, 0.03529411764705882, 0.9999999716771969, 0.5882352941176471, 0.11372549019607843, 0.023529411764705882, 0.9999999728874311, 0.6627450980392157, 0.16862745098039217, 0.01568627450980392, 0.9999999740976654, 0.7411764705882353, 0.22745098039215686, 0.00392156862745098, 0.9999999753079026, 0.788235294117647, 0.2901960784313726, 0.0, 0.9999999765181369, 0.8627450980392157, 0.3803921568627451, 0.011764705882352941, 0.9999999777283711, 0.9019607843137255, 0.4588235294117647, 0.027450980392156862, 0.9999999789386084, 0.9176470588235294, 0.5215686274509804, 0.047058823529411764, 0.9999999801488426, 0.9254901960784314, 0.5803921568627451, 0.0784313725490196, 0.9999999813590769, 0.9372549019607843, 0.6431372549019608, 0.12156862745098039, 0.9999999825693112, 0.9450980392156862, 0.7098039215686275, 0.1843137254901961, 0.9999999837795484, 0.9529411764705882, 0.7686274509803922, 0.24705882352941178, 0.9999999849897826, 0.9647058823529412, 0.8274509803921568, 0.3254901960784314, 0.999999986200017, 0.9686274509803922, 0.8784313725490196, 0.4235294117647059, 0.9999999874102512, 0.9725490196078431, 0.9176470588235294, 0.5137254901960784, 0.9999999886204883, 0.9803921568627451, 0.9490196078431372, 0.596078431372549, 0.9999999898307227, 0.9803921568627451, 0.9725490196078431, 0.6705882352941176, 0.9999999910409569, 0.9882352941176471, 0.9882352941176471, 0.7568627450980392, 0.9999999922027831, 0.984313725490196, 0.9882352941176471, 0.8549019607843137, 0.9999999922511941, 0.9882352941176471, 0.9882352941176471, 0.8588235294117647, 0.9999999922519326, 0.9529411764705882, 0.9529411764705882, 0.8941176470588236, 0.9999999922519326, 0.9529411764705882, 0.9529411764705882, 0.8941176470588236, 0.9999999935087229, 0.8901960784313725, 0.8901960784313725, 0.807843137254902, 0.9999999947655162, 0.8274509803921568, 0.8235294117647058, 0.7372549019607844, 0.9999999960223094, 0.7764705882352941, 0.7647058823529411, 0.6784313725490196, 0.9999999972791026, 0.7254901960784313, 0.7137254901960784, 0.6274509803921569, 0.9999999985358958, 0.6784313725490196, 0.6627450980392157, 0.5803921568627451, 0.9999999997926862, 0.6313725490196078, 0.6078431372549019, 0.5333333333333333, 1.0000000010494794, 0.5803921568627451, 0.5568627450980392, 0.48627450980392156, 1.0000000023062725, 0.5372549019607843, 0.5058823529411764, 0.44313725490196076, 1.0000000035630658, 0.4980392156862745, 0.4588235294117647, 0.40784313725490196, 1.0000000048198592, 0.4627450980392157, 0.4196078431372549, 0.37254901960784315, 1.0000000060766494, 0.43137254901960786, 0.38823529411764707, 0.34509803921568627, 1.0000000073334425, 0.403921568627451, 0.3568627450980392, 0.3176470588235294, 1.0000000085902359, 0.37254901960784315, 0.3215686274509804, 0.29411764705882354, 1.0000000098470292, 0.34509803921568627, 0.29411764705882354, 0.26666666666666666, 1.0000000111038223, 0.3176470588235294, 0.2627450980392157, 0.23921568627450981, 1.0000000123606125, 0.28627450980392155, 0.23137254901960785, 0.21176470588235294, 1.0000000136174059, 0.2549019607843137, 0.2, 0.1843137254901961, 1.0000000148741992, 0.23137254901960785, 0.17254901960784313, 0.16470588235294117, 1.0000000161309923, 0.2, 0.1450980392156863, 0.13725490196078433, 1.000000017388524, 0.14902, 0.196078, 0.278431, 1.0000000198381944, 0.2, 0.2549019607843137, 0.34509803921568627, 1.0000000222878649, 0.24705882352941178, 0.3176470588235294, 0.41568627450980394, 1.0000000247375354, 0.3058823529411765, 0.38823529411764707, 0.49411764705882355, 1.0000000271872058, 0.37254901960784315, 0.4588235294117647, 0.5686274509803921, 1.0000000296368763, 0.44313725490196076, 0.5333333333333333, 0.6431372549019608, 1.0000000320865468, 0.5176470588235295, 0.615686274509804, 0.7254901960784313, 1.0000000345362172, 0.6, 0.6980392156862745, 0.8, 1.0000000369858877, 0.6862745098039216, 0.7843137254901961, 0.8705882352941177, 1.0000000394355582, 0.7607843137254902, 0.8588235294117647, 0.9294117647058824, 1.000000040660392, 0.807843137254902, 0.9019607843137255, 0.9607843137254902, 1.0000000418852286, 0.8901960784313725, 0.9568627450980393, 0.984313725490196]
normalsLUT.ColorSpace = 'Lab'
normalsLUT.NanColor = [1.0, 0.0, 0.0]
normalsLUT.ScalarRangeInitialized = 1.0

# get color legend/bar for normalsLUT in view renderView1
normalsLUTColorBar = GetScalarBar(normalsLUT, renderView1)
normalsLUTColorBar.WindowLocation = 'Upper Right Corner'
normalsLUTColorBar.Title = 'Normals'
normalsLUTColorBar.ComponentTitle = 'Magnitude'

# set color bar visibility
normalsLUTColorBar.Visibility = 0

# get color legend/bar for separate_sphere1Display_NormalsLUT in view renderView1
separate_sphere1Display_NormalsLUTColorBar = GetScalarBar(separate_sphere1Display_NormalsLUT, renderView1)
separate_sphere1Display_NormalsLUTColorBar.WindowLocation = 'Upper Right Corner'
separate_sphere1Display_NormalsLUTColorBar.Title = 'Normals'
separate_sphere1Display_NormalsLUTColorBar.ComponentTitle = 'Z'

# set color bar visibility
separate_sphere1Display_NormalsLUTColorBar.Visibility = 1

# show color legend
tde_snapshot00000h5Display.SetScalarBarVisibility(renderView1, True)

# hide data in view
Hide(tde_snapshot00000h5, renderView1)

# show color legend
pointVolumeInterpolator1Display.SetScalarBarVisibility(renderView1, True)

# show color legend
sphere1Display.SetScalarBarVisibility(renderView1, True)

# ----------------------------------------------------------------
# setup color maps and opacity mapes used in the visualization
# note: the Get..() functions create a new object, if needed
# ----------------------------------------------------------------

# get separate opacity transfer function/opacity map for 'rho'
separate_threshold1Display_rhoPWF = GetOpacityTransferFunction('rho', threshold1Display, separate=True)
separate_threshold1Display_rhoPWF.Points = [1.5612487302973932e-08, 0.0, 0.5, 0.0, 1.9999947653559502e-06, 1.0, 0.5, 0.0]
separate_threshold1Display_rhoPWF.ScalarRangeInitialized = 1

# get separate opacity transfer function/opacity map for 'Normals'
separate_sphere1Display_NormalsPWF = GetOpacityTransferFunction('Normals', sphere1Display, separate=True)
separate_sphere1Display_NormalsPWF.Points = [-1.0127336778678, 0.6312500238418579, 0.5, 0.0, -0.04383107274770737, 0.3062500059604645, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]
separate_sphere1Display_NormalsPWF.ScalarRangeInitialized = 1

# get opacity transfer function/opacity map for 'Normals'
normalsPWF = GetOpacityTransferFunction('Normals')
normalsPWF.Points = [0.9999999680464912, 0.0, 0.5, 0.0, 1.0000000418852286, 1.0, 0.5, 0.0]
normalsPWF.ScalarRangeInitialized = 1

# ----------------------------------------------------------------
# setup extractors
# ----------------------------------------------------------------

# create extractor
pNG2 = CreateExtractor('PNG', renderView1, registrationName='PNG2')
# trace defaults for the extractor.
pNG2.Trigger = 'TimeStep'

# init the 'TimeStep' selected for 'Trigger'
pNG2.Trigger.UseStartTimeStep = 1
pNG2.Trigger.UseEndTimeStep = 1
pNG2.Trigger.EndTimeStep = 300

# init the 'PNG' selected for 'Writer'
pNG2.Writer.FileName = 'RenderView1_{timestep:06d}{camera}.png'
pNG2.Writer.ImageResolution = [2427, 1321]
pNG2.Writer.Format = 'PNG'

# ----------------------------------------------------------------
# restore active source
SetActiveSource(threshold1)
# ----------------------------------------------------------------


if __name__ == '__main__':
    # generate extracts
    SaveExtracts(ExtractsOutputDirectory='/home/appcell/extracts/')
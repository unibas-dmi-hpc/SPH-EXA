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

# a texture
skybox = CreateTexture('/home/appcell/Visualizations/TidalDisruptionEvent/Datasets/skybox.jpg')

# Create a new 'Render View'
renderView1 = CreateView('RenderView')
renderView1.ViewSize = [2427, 1422]
renderView1.AxesGrid = 'GridAxes3DActor'
renderView1.CenterOfRotation = [-141.15701293945312, -188.40384674072266, 0.0008760541677474976]
renderView1.StereoType = 'Crystal Eyes'
renderView1.CameraPosition = [-140.9743050567837, -189.46470744907455, -2.7993632870741294]
renderView1.CameraFocalPoint = [-141.1570129394531, -188.40384674072266, 0.0008760541677194824]
renderView1.CameraViewUp = [-0.8005244251547518, 0.5412615255790755, -0.257287010279311]
renderView1.CameraFocalDisk = 1.0
renderView1.CameraParallelScale = 0.7764635186713663
renderView1.UseColorPaletteForBackground = 0
renderView1.BackgroundColorMode = 'Stereo Skybox'
renderView1.BackgroundTexture = skybox

SetActiveView(None)

# ----------------------------------------------------------------
# setup view layouts
# ----------------------------------------------------------------

# create new layout object 'Layout #1'
layout1 = CreateLayout(name='Layout #1')
layout1.AssignView(0, renderView1)
layout1.SetSize(2427, 1422)

# ----------------------------------------------------------------
# restore active view
SetActiveView(renderView1)
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# setup the data processing pipelines
# ----------------------------------------------------------------

# create a new 'Sphere'
sphere1 = Sphere(registrationName='Sphere1')
sphere1.Radius = 15.0
sphere1.ThetaResolution = 32
sphere1.PhiResolution = 32

# create a new 'H5PartReader'
tde_snapshot00000h5 = H5PartReader(registrationName='tde_snapshot00000.h5', FileName='/home/appcell/Visualizations/TidalDisruptionEvent/Datasets/tde_snapshot00000.h5')
tde_snapshot00000h5.Xarray = 'x'
tde_snapshot00000h5.Yarray = 'y'
tde_snapshot00000h5.Zarray = 'z'
tde_snapshot00000h5.PointArrays = ['rho', 'x', 'y', 'z']

# create a new 'Point Source'
pointSource1 = PointSource(registrationName='PointSource1')

# create a new 'Point Volume Interpolator'
pointVolumeInterpolator1 = PointVolumeInterpolator(registrationName='PointVolumeInterpolator1', Input=tde_snapshot00000h5,
    Source='Bounded Volume')
pointVolumeInterpolator1.Kernel = 'VoronoiKernel'
pointVolumeInterpolator1.Locator = 'Static Point Locator'

# init the 'Bounded Volume' selected for 'Source'
pointVolumeInterpolator1.Source.Origin = [-141.60470581054688, -188.85299682617188, -0.4471539556980133]
pointVolumeInterpolator1.Source.Scale = [0.8953857421875, 0.8983001708984375, 0.8960600197315216]

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
rhoLUT.AutomaticRescaleRangeMode = 'Grow and update every timestep'
rhoLUT.TransferFunction2D = rhoTF2D
rhoLUT.RGBPoints = [9.366737430127614e-08, 0.001462, 0.000466, 0.013866, 1.598293400924291e-07, 0.002267, 0.00127, 0.01857, 2.2597443643794433e-07, 0.003299, 0.002249, 0.024239, 2.921364022290973e-07, 0.004547, 0.003392, 0.030909, 3.582814985746125e-07, 0.006006, 0.004692, 0.038558, 4.2444346436576553e-07, 0.007676, 0.006136, 0.046836, 4.905885607112808e-07, 0.009561, 0.007713, 0.055143, 5.567505265024337e-07, 0.011663, 0.009417, 0.06346, 6.229124922935867e-07, 0.013995, 0.011225, 0.071862, 6.890575886391019e-07, 0.016561, 0.013136, 0.080282, 7.552195544302549e-07, 0.019373, 0.015133, 0.088767, 8.213646507757701e-07, 0.022447, 0.017199, 0.097327, 8.87526616566923e-07, 0.025793, 0.019331, 0.10593, 9.536717129124383e-07, 0.029432, 0.021503, 0.114621, 1.0198336787035912e-06, 0.033385, 0.023702, 0.123397, 1.0859956444947443e-06, 0.037668, 0.025921, 0.132232, 1.1521407408402594e-06, 0.042253, 0.028139, 0.141141, 1.2183027066314126e-06, 0.046915, 0.030324, 0.150164, 1.2844478029769277e-06, 0.051644, 0.032474, 0.159254, 1.350609768768081e-06, 0.056449, 0.034569, 0.168414, 1.416754865113596e-06, 0.06134, 0.03659, 0.177642, 1.4829168309047488e-06, 0.066331, 0.038504, 0.186962, 1.549078796695902e-06, 0.071429, 0.040294, 0.196354, 1.6152238930414171e-06, 0.076637, 0.041905, 0.205799, 1.68138585883257e-06, 0.081962, 0.043328, 0.215289, 1.7475309551780852e-06, 0.087411, 0.044556, 0.224813, 1.8136929209692382e-06, 0.09299, 0.045583, 0.234358, 1.8798380173147536e-06, 0.098702, 0.046402, 0.243904, 1.9459999831059064e-06, 0.104551, 0.047008, 0.25343, 2.012145079451422e-06, 0.110536, 0.047399, 0.262912, 2.0783070452425747e-06, 0.116656, 0.047574, 0.272321, 2.1444690110337275e-06, 0.122908, 0.047536, 0.281624, 2.2106141073792426e-06, 0.129285, 0.047293, 0.290788, 2.276776073170396e-06, 0.135778, 0.046856, 0.299776, 2.3429211695159113e-06, 0.142378, 0.046242, 0.308553, 2.4090831353070637e-06, 0.149073, 0.045468, 0.317085, 2.4752282316525792e-06, 0.15585, 0.044559, 0.325338, 2.5413901974437324e-06, 0.162689, 0.043554, 0.333277, 2.6075521632348857e-06, 0.169575, 0.042489, 0.340874, 2.6736972595804003e-06, 0.176493, 0.041402, 0.348111, 2.7398592253715536e-06, 0.183429, 0.040329, 0.354971, 2.8060043217170687e-06, 0.190367, 0.039309, 0.361447, 2.8721662875082215e-06, 0.197297, 0.0384, 0.367535, 2.938311383853737e-06, 0.204209, 0.037632, 0.373238, 3.00447334964489e-06, 0.211095, 0.03703, 0.378563, 3.0706353154360426e-06, 0.217949, 0.036615, 0.383522, 3.136780411781558e-06, 0.224763, 0.036405, 0.388129, 3.2029423775727113e-06, 0.231538, 0.036405, 0.3924, 3.2690874739182264e-06, 0.238273, 0.036621, 0.396353, 3.3352494397093792e-06, 0.244967, 0.037055, 0.400007, 3.4013945360548943e-06, 0.25162, 0.037705, 0.403378, 3.4675565018460476e-06, 0.258234, 0.038571, 0.406485, 3.5337184676372003e-06, 0.26481, 0.039647, 0.409345, 3.599863563982716e-06, 0.271347, 0.040922, 0.411976, 3.6660255297738687e-06, 0.27785, 0.042353, 0.414392, 3.7321706261193838e-06, 0.284321, 0.043933, 0.416608, 3.798332591910537e-06, 0.290763, 0.045644, 0.418637, 3.8644776882560525e-06, 0.297178, 0.04747, 0.420491, 3.9306396540472045e-06, 0.303568, 0.049396, 0.422182, 3.996801619838358e-06, 0.309935, 0.051407, 0.423721, 4.062946716183873e-06, 0.316282, 0.05349, 0.425116, 4.129108681975026e-06, 0.32261, 0.055634, 0.426377, 4.195253778320541e-06, 0.328921, 0.057827, 0.427511, 4.261415744111695e-06, 0.335217, 0.06006, 0.428524, 4.327560840457209e-06, 0.3415, 0.062325, 0.429425, 4.393722806248363e-06, 0.347771, 0.064616, 0.430217, 4.4598847720395155e-06, 0.354032, 0.066925, 0.430906, 4.5260298683850306e-06, 0.360284, 0.069247, 0.431497, 4.592191834176183e-06, 0.366529, 0.071579, 0.431994, 4.658336930521699e-06, 0.372768, 0.073915, 0.4324, 4.724498896312851e-06, 0.379001, 0.076253, 0.432719, 4.790643992658367e-06, 0.385228, 0.078591, 0.432955, 4.856805958449521e-06, 0.391453, 0.080927, 0.433109, 4.922967924240674e-06, 0.397674, 0.083257, 0.433183, 4.989113020586189e-06, 0.403894, 0.08558, 0.433179, 5.0552749863773415e-06, 0.410113, 0.087896, 0.433098, 5.121420082722857e-06, 0.416331, 0.090203, 0.432943, 5.1875820485140094e-06, 0.422549, 0.092501, 0.432714, 5.2537271448595246e-06, 0.428768, 0.09479, 0.432412, 5.319889110650678e-06, 0.434987, 0.097069, 0.432039, 5.3860342069961925e-06, 0.441207, 0.099338, 0.431594, 5.452196172787346e-06, 0.447428, 0.101597, 0.43108, 5.518358138578499e-06, 0.453651, 0.103848, 0.430498, 5.584503234924014e-06, 0.459875, 0.106089, 0.429846, 5.650665200715167e-06, 0.4661, 0.108322, 0.429125, 5.716810297060682e-06, 0.472328, 0.110547, 0.428334, 5.7829722628518355e-06, 0.478558, 0.112764, 0.427475, 5.849117359197351e-06, 0.484789, 0.114974, 0.426548, 5.915279324988504e-06, 0.491022, 0.117179, 0.425552, 5.981441290779656e-06, 0.497257, 0.119379, 0.424488, 6.047586387125172e-06, 0.503493, 0.121575, 0.423356, 6.113748352916325e-06, 0.50973, 0.123769, 0.422156, 6.17989344926184e-06, 0.515967, 0.12596, 0.420887, 6.246055415052993e-06, 0.522206, 0.12815, 0.419549, 6.312200511398508e-06, 0.528444, 0.130341, 0.418142, 6.378362477189662e-06, 0.534683, 0.132534, 0.416667, 6.444524442980814e-06, 0.54092, 0.134729, 0.415123, 6.5106695393263295e-06, 0.547157, 0.136929, 0.413511, 6.576831505117482e-06, 0.553392, 0.139134, 0.411829, 6.642976601462997e-06, 0.559624, 0.141346, 0.410078, 6.70913856725415e-06, 0.565854, 0.143567, 0.408258, 6.775283663599665e-06, 0.572081, 0.145797, 0.406369, 6.841445629390819e-06, 0.578304, 0.148039, 0.404411, 6.907607595181972e-06, 0.584521, 0.150294, 0.402385, 6.973752691527487e-06, 0.590734, 0.152563, 0.40029, 7.03991465731864e-06, 0.59694, 0.154848, 0.398125, 7.106059753664156e-06, 0.603139, 0.157151, 0.395891, 7.1722217194553076e-06, 0.60933, 0.159474, 0.393589, 7.2383668158008235e-06, 0.615513, 0.161817, 0.391219, 7.304528781591977e-06, 0.621685, 0.164184, 0.388781, 7.370690747383129e-06, 0.627847, 0.166575, 0.386276, 7.436835843728645e-06, 0.633998, 0.168992, 0.383704, 7.502997809519798e-06, 0.640135, 0.171438, 0.381065, 7.569142905865313e-06, 0.64626, 0.173914, 0.378359, 7.635304871656465e-06, 0.652369, 0.176421, 0.375586, 7.701449968001981e-06, 0.658463, 0.178962, 0.372748, 7.767611933793133e-06, 0.66454, 0.181539, 0.369846, 7.833773899584286e-06, 0.670599, 0.184153, 0.366879, 7.899918995929802e-06, 0.676638, 0.186807, 0.363849, 7.966080961720954e-06, 0.682656, 0.189501, 0.360757, 8.03222605806647e-06, 0.688653, 0.192239, 0.357603, 8.098388023857624e-06, 0.694627, 0.195021, 0.354388, 8.164533120203138e-06, 0.700576, 0.197851, 0.351113, 8.230695085994292e-06, 0.7065, 0.200728, 0.347777, 8.296857051785445e-06, 0.712396, 0.203656, 0.344383, 8.36300214813096e-06, 0.718264, 0.206636, 0.340931, 8.429164113922113e-06, 0.724103, 0.20967, 0.337424, 8.495309210267628e-06, 0.729909, 0.212759, 0.333861, 8.561471176058781e-06, 0.735683, 0.215906, 0.330245, 8.627616272404297e-06, 0.741423, 0.219112, 0.326576, 8.69377823819545e-06, 0.747127, 0.222378, 0.322856, 8.759923334540963e-06, 0.752794, 0.225706, 0.319085, 8.826085300332117e-06, 0.758422, 0.229097, 0.315266, 8.892247266123269e-06, 0.76401, 0.232554, 0.311399, 8.958392362468785e-06, 0.769556, 0.236077, 0.307485, 9.024554328259939e-06, 0.775059, 0.239667, 0.303526, 9.090699424605453e-06, 0.780517, 0.243327, 0.299523, 9.156861390396608e-06, 0.785929, 0.247056, 0.295477, 9.223006486742122e-06, 0.791293, 0.250856, 0.29139, 9.289168452533274e-06, 0.796607, 0.254728, 0.287264, 9.355330418324426e-06, 0.801871, 0.258674, 0.283099, 9.421475514669944e-06, 0.807082, 0.262692, 0.278898, 9.487637480461096e-06, 0.812239, 0.266786, 0.274661, 9.55378257680661e-06, 0.817341, 0.270954, 0.27039, 9.619944542597766e-06, 0.822386, 0.275197, 0.266085, 9.68608963894328e-06, 0.827372, 0.279517, 0.26175, 9.752251604734432e-06, 0.832299, 0.283913, 0.257383, 9.818413570525585e-06, 0.837165, 0.288385, 0.252988, 9.884558666871101e-06, 0.841969, 0.292933, 0.248564, 9.950720632662253e-06, 0.846709, 0.297559, 0.244113, 1.0016865729007768e-05, 0.851384, 0.30226, 0.239636, 1.0083027694798923e-05, 0.855992, 0.307038, 0.235133, 1.0149172791144437e-05, 0.860533, 0.311892, 0.230606, 1.0215334756935589e-05, 0.865006, 0.316822, 0.226055, 1.0281496722726743e-05, 0.869409, 0.321827, 0.221482, 1.0347641819072259e-05, 0.873741, 0.326906, 0.216886, 1.041380378486341e-05, 0.878001, 0.33206, 0.212268, 1.0479948881208927e-05, 0.882188, 0.337287, 0.207628, 1.054611084700008e-05, 0.886302, 0.342586, 0.202968, 1.0612255943345595e-05, 0.890341, 0.347957, 0.198286, 1.0678417909136748e-05, 0.894305, 0.353399, 0.193584, 1.07445798749279e-05, 0.898192, 0.358911, 0.18886, 1.0810724971273416e-05, 0.902003, 0.364492, 0.184116, 1.087688693706457e-05, 0.905735, 0.37014, 0.17935, 1.0943032033410084e-05, 0.90939, 0.375856, 0.174563, 1.1009193999201238e-05, 0.912966, 0.381636, 0.169755, 1.1075339095546752e-05, 0.916462, 0.387481, 0.164924, 1.1141501061337905e-05, 0.919879, 0.393389, 0.16007, 1.1207663027129057e-05, 0.923215, 0.399359, 0.155193, 1.1273808123474573e-05, 0.92647, 0.405389, 0.150292, 1.1339970089265727e-05, 0.929644, 0.411479, 0.145367, 1.1406115185611241e-05, 0.932737, 0.417627, 0.140417, 1.1472277151402395e-05, 0.935747, 0.423831, 0.13544, 1.153842224774791e-05, 0.938675, 0.430091, 0.130438, 1.1604584213539063e-05, 0.941521, 0.436405, 0.125409, 1.1670746179330215e-05, 0.944285, 0.442772, 0.120354, 1.1736891275675732e-05, 0.946965, 0.449191, 0.115272, 1.1803053241466884e-05, 0.949562, 0.45566, 0.110164, 1.1869198337812399e-05, 0.952075, 0.462178, 0.105031, 1.193536030360355e-05, 0.954506, 0.468744, 0.099874, 1.2001505399949068e-05, 0.956852, 0.475356, 0.094695, 1.206766736574022e-05, 0.959114, 0.482014, 0.089499, 1.2133812462085736e-05, 0.961293, 0.488716, 0.084289, 1.219997442787689e-05, 0.963387, 0.495462, 0.079073, 1.2266136393668042e-05, 0.965397, 0.502249, 0.073859, 1.2332281490013556e-05, 0.967322, 0.509078, 0.068659, 1.239844345580471e-05, 0.969163, 0.515946, 0.063488, 1.2464588552150226e-05, 0.970919, 0.522853, 0.058367, 1.2530750517941378e-05, 0.97259, 0.529798, 0.053324, 1.2596895614286892e-05, 0.974176, 0.53678, 0.048392, 1.2663057580078047e-05, 0.975677, 0.543798, 0.043618, 1.2729219545869199e-05, 0.977092, 0.55085, 0.03905, 1.2795364642214713e-05, 0.978422, 0.557937, 0.034931, 1.2861526608005867e-05, 0.979666, 0.565057, 0.031409, 1.2927671704351383e-05, 0.980824, 0.572209, 0.028508, 1.2993833670142535e-05, 0.981895, 0.579392, 0.02625, 1.305997876648805e-05, 0.982881, 0.586606, 0.024661, 1.3126140732279204e-05, 0.983779, 0.593849, 0.02377, 1.3192302698070356e-05, 0.984591, 0.601122, 0.023606, 1.3258447794415872e-05, 0.985315, 0.608422, 0.024202, 1.3324609760207024e-05, 0.985952, 0.61575, 0.025592, 1.339075485655254e-05, 0.986502, 0.623105, 0.027814, 1.3456916822343694e-05, 0.986964, 0.630485, 0.030908, 1.3523061918689208e-05, 0.987337, 0.63789, 0.034916, 1.3589223884480362e-05, 0.987622, 0.64532, 0.039886, 1.3655385850271515e-05, 0.987819, 0.652773, 0.045581, 1.372153094661703e-05, 0.987926, 0.66025, 0.05175, 1.3787692912408182e-05, 0.987945, 0.667748, 0.058329, 1.3853838008753698e-05, 0.987874, 0.675267, 0.065257, 1.3919999974544851e-05, 0.987714, 0.682807, 0.072489, 1.3986145070890365e-05, 0.987464, 0.690366, 0.07999, 1.4052307036681519e-05, 0.987124, 0.697944, 0.087731, 1.4118469002472673e-05, 0.986694, 0.70554, 0.095694, 1.4184614098818187e-05, 0.986175, 0.713153, 0.103863, 1.4250776064609339e-05, 0.985566, 0.720782, 0.112229, 1.4316921160954857e-05, 0.984865, 0.728427, 0.120785, 1.4383083126746009e-05, 0.984075, 0.736087, 0.129527, 1.4449228223091523e-05, 0.983196, 0.743758, 0.138453, 1.4515390188882678e-05, 0.982228, 0.751442, 0.147565, 1.458155215467383e-05, 0.981173, 0.759135, 0.156863, 1.4647697251019344e-05, 0.980032, 0.766837, 0.166353, 1.4713859216810498e-05, 0.978806, 0.774545, 0.176037, 1.4780004313156014e-05, 0.977497, 0.782258, 0.185923, 1.4846166278947166e-05, 0.976108, 0.789974, 0.196018, 1.491231137529268e-05, 0.974638, 0.797692, 0.206332, 1.4978473341083836e-05, 0.973088, 0.805409, 0.216877, 1.5044635306874987e-05, 0.971468, 0.813122, 0.227658, 1.5110780403220502e-05, 0.969783, 0.820825, 0.238686, 1.5176942369011655e-05, 0.968041, 0.828515, 0.249972, 1.5243087465357171e-05, 0.966243, 0.836191, 0.261534, 1.530924943114832e-05, 0.964394, 0.843848, 0.273391, 1.537539452749384e-05, 0.962517, 0.851476, 0.285546, 1.544155649328499e-05, 0.960626, 0.859069, 0.29801, 1.5507701589630507e-05, 0.95872, 0.866624, 0.31082, 1.557386355542166e-05, 0.956834, 0.874129, 0.323974, 1.564002552121281e-05, 0.954997, 0.881569, 0.337475, 1.570617061755833e-05, 0.953215, 0.888942, 0.351369, 1.5772332583349482e-05, 0.951546, 0.896226, 0.365627, 1.5838477679694997e-05, 0.950018, 0.903409, 0.380271, 1.590463964548615e-05, 0.948683, 0.910473, 0.395289, 1.5970784741831664e-05, 0.947594, 0.917399, 0.410665, 1.6036946707622818e-05, 0.946809, 0.924168, 0.426373, 1.6103108673413972e-05, 0.946392, 0.930761, 0.442367, 1.6169253769759486e-05, 0.946403, 0.937159, 0.458592, 1.623541573555064e-05, 0.946903, 0.943348, 0.47497, 1.6301560831896154e-05, 0.947937, 0.949318, 0.491426, 1.6367722797687308e-05, 0.949545, 0.955063, 0.50786, 1.6433867894032822e-05, 0.95174, 0.960587, 0.524203, 1.6500029859823975e-05, 0.954529, 0.965896, 0.540361, 1.656619182561513e-05, 0.957896, 0.971003, 0.556275, 1.6632336921960643e-05, 0.961812, 0.975924, 0.571925, 1.6698498887751797e-05, 0.966249, 0.980678, 0.587206, 1.676464398409731e-05, 0.971162, 0.985282, 0.602154, 1.6830805949888465e-05, 0.976511, 0.989753, 0.61676, 1.689695104623398e-05, 0.982257, 0.994109, 0.631017, 1.6963113012025133e-05, 0.988362, 0.998364, 0.644924]
rhoLUT.NanColor = [0.0, 1.0, 0.0]
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
rhoPWF.Points = [9.366737430127614e-08, 0.0, 0.5, 0.0, 1.9560193322831765e-06, 0.02717391401529312, 0.5, 0.0, 2.3674692783970386e-06, 0.125, 0.5, 0.0, 4.48968467026134e-06, 0.043478261679410934, 0.5, 0.0, 9.860187674348708e-06, 0.04891304299235344, 0.5, 0.0, 9.860187674348708e-06, 0.05434782803058624, 0.5, 0.0, 1.3541582120524254e-05, 0.20652174949645996, 0.5, 0.0, 1.6963113012025133e-05, 1.0, 0.5, 0.0]
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
pointVolumeInterpolator1Display.IsosurfaceValues = [8.00048490745553e-06]
pointVolumeInterpolator1Display.SliceFunction = 'Plane'
pointVolumeInterpolator1Display.Slice = 50
pointVolumeInterpolator1Display.SelectInputVectors = [None, '']
pointVolumeInterpolator1Display.WriteLog = ''

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
pointVolumeInterpolator1Display.ScaleTransferFunction.Points = [9.366737430127614e-08, 0.0, 0.5, 0.0, 1.5907302440609783e-05, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
pointVolumeInterpolator1Display.OpacityTransferFunction.Points = [9.366737430127614e-08, 0.0, 0.5, 0.0, 1.5907302440609783e-05, 1.0, 0.5, 0.0]

# init the 'Plane' selected for 'SliceFunction'
pointVolumeInterpolator1Display.SliceFunction.Origin = [-141.15701293945312, -188.40384674072266, 0.000876054167747442]

# show data from sphere1
sphere1Display = Show(sphere1, renderView1, 'GeometryRepresentation')

# trace defaults for the display properties.
sphere1Display.Representation = 'Surface'
sphere1Display.AmbientColor = [0.0, 0.0, 0.0]
sphere1Display.ColorArrayName = ['POINTS', '']
sphere1Display.DiffuseColor = [0.0, 0.0, 0.0]
sphere1Display.Opacity = 0.31
sphere1Display.Specular = 0.06
sphere1Display.SelectTCoordArray = 'None'
sphere1Display.SelectNormalArray = 'Normals'
sphere1Display.SelectTangentArray = 'None'
sphere1Display.OSPRayScaleArray = 'Normals'
sphere1Display.OSPRayScaleFunction = 'PiecewiseFunction'
sphere1Display.SelectOrientationVectors = 'None'
sphere1Display.ScaleFactor = 4.0
sphere1Display.SelectScaleArray = 'None'
sphere1Display.GlyphType = 'Arrow'
sphere1Display.GlyphTableIndexArray = 'None'
sphere1Display.GaussianRadius = 0.2
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

# show data from pointSource1
pointSource1Display = Show(pointSource1, renderView1, 'GeometryRepresentation')

# trace defaults for the display properties.
pointSource1Display.Representation = 'Point Gaussian'
pointSource1Display.AmbientColor = [0.0, 0.0, 0.0]
pointSource1Display.ColorArrayName = [None, '']
pointSource1Display.DiffuseColor = [0.0, 0.0, 0.0]
pointSource1Display.Opacity = 0.71
pointSource1Display.SelectTCoordArray = 'None'
pointSource1Display.SelectNormalArray = 'None'
pointSource1Display.SelectTangentArray = 'None'
pointSource1Display.OSPRayScaleFunction = 'PiecewiseFunction'
pointSource1Display.SelectOrientationVectors = 'None'
pointSource1Display.ScaleFactor = 0.1
pointSource1Display.SelectScaleArray = 'None'
pointSource1Display.GlyphType = 'Arrow'
pointSource1Display.GlyphTableIndexArray = 'None'
pointSource1Display.GaussianRadius = 10.0
pointSource1Display.SetScaleArray = [None, '']
pointSource1Display.ScaleTransferFunction = 'PiecewiseFunction'
pointSource1Display.OpacityArray = [None, '']
pointSource1Display.OpacityTransferFunction = 'PiecewiseFunction'
pointSource1Display.DataAxesGrid = 'GridAxesRepresentation'
pointSource1Display.PolarAxes = 'PolarAxesRepresentation'
pointSource1Display.SelectInputVectors = [None, '']
pointSource1Display.WriteLog = ''

# setup the color legend parameters for each legend in this view

# get color legend/bar for rhoLUT in view renderView1
rhoLUTColorBar = GetScalarBar(rhoLUT, renderView1)
rhoLUTColorBar.Title = 'rho'
rhoLUTColorBar.ComponentTitle = ''

# set color bar visibility
rhoLUTColorBar.Visibility = 1

# get 2D transfer function for 'Normals'
normalsTF2D = GetTransferFunction2D('Normals')

# get color transfer function/color map for 'Normals'
normalsLUT = GetColorTransferFunction('Normals')
normalsLUT.TransferFunction2D = normalsTF2D
normalsLUT.RGBPoints = [-0.9987165331840515, 0.0, 0.0, 0.0, -0.8733875968362581, 0.054902, 0.054902, 0.075817, -0.7480586574851683, 0.109804, 0.109804, 0.151634, -0.6227287210396573, 0.164706, 0.164706, 0.227451, -0.49739978469186397, 0.219608, 0.219608, 0.303268, -0.37207084534077417, 0.27451, 0.27451, 0.379085, -0.24674190899298087, 0.329412, 0.329902, 0.454412, -0.12141297264518747, 0.384314, 0.405719, 0.509314, 0.003916534328931354, 0.439216, 0.481536, 0.564216, 0.1292459031514135, 0.494118, 0.557353, 0.619118, 0.2545748394992067, 0.54902, 0.63317, 0.67402, 0.37990378185359286, 0.603922, 0.708987, 0.728922, 0.5052327182013863, 0.660294, 0.783824, 0.783824, 0.6305626516436009, 0.746569, 0.838725, 0.838725, 0.7558915939979873, 0.832843, 0.893627, 0.893627, 0.8812205303457805, 0.919118, 0.948529, 0.948529, 0.9987165331840515, 1.0, 1.0, 1.0]
normalsLUT.ColorSpace = 'Lab'
normalsLUT.NanColor = [0.25, 0.0, 0.0]
normalsLUT.ScalarRangeInitialized = 1.0
normalsLUT.VectorMode = 'Component'

# get color legend/bar for normalsLUT in view renderView1
normalsLUTColorBar = GetScalarBar(normalsLUT, renderView1)
normalsLUTColorBar.WindowLocation = 'Upper Right Corner'
normalsLUTColorBar.Title = 'Normals'
normalsLUTColorBar.ComponentTitle = 'X'

# set color bar visibility
normalsLUTColorBar.Visibility = 0

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

# get opacity transfer function/opacity map for 'Normals'
normalsPWF = GetOpacityTransferFunction('Normals')
normalsPWF.Points = [-0.9987165331840515, 0.0, 0.5, 0.0, 0.9987165331840515, 1.0, 0.5, 0.0]
normalsPWF.ScalarRangeInitialized = 1

# ----------------------------------------------------------------
# setup extractors
# ----------------------------------------------------------------

# create extractor
jPG1 = CreateExtractor('JPG', renderView1, registrationName='JPG1')
# trace defaults for the extractor.
jPG1.Trigger = 'TimeStep'

# init the 'JPG' selected for 'Writer'
jPG1.Writer.FileName = 'RenderView1_{timestep:06d}{camera}.jpg'
jPG1.Writer.ImageResolution = [2427, 1422]
jPG1.Writer.Format = 'JPEG'

# ----------------------------------------------------------------
# restore active source
SetActiveSource(jPG1)
# ----------------------------------------------------------------


if __name__ == '__main__':
    # generate extracts
    SaveExtracts(ExtractsOutputDirectory='/home/appcell/extracts')
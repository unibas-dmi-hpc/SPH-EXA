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

# create light
# Create a new 'Render View'
renderView1 = CreateView('RenderView')
renderView1.ViewSize = [2427, 1321]
renderView1.AxesGrid = 'GridAxes3DActor'
renderView1.OrientationAxesVisibility = 0
renderView1.CenterOfRotation = [-141.15701293945312, -188.40384674072266, 0.0008760541677474976]
renderView1.KeyLightIntensity = 1.0
renderView1.FillLightWarmth = 0.9999999999999999
renderView1.FillLightKFRatio = 5.5
renderView1.BackLightKBRatio = 2.0
renderView1.StereoType = 'Crystal Eyes'
renderView1.CameraPosition = [-69.85199356079102, -148.18453979492188, 24.42368398006215]
renderView1.CameraFocalPoint = [-69.85199356079102, -148.18453979492188, 0.0026583969593048096]
renderView1.CameraFocalDisk = 1.0
renderView1.CameraParallelScale = 0.7764635186713663
renderView1.UseColorPaletteForBackground = 0
renderView1.Background = [0.05888456549935149, 0.08593881132219425, 0.17666895551995118]
renderView1.AdditionalLights = light1

SetActiveView(None)

# ----------------------------------------------------------------
# setup view layouts
# ----------------------------------------------------------------

# create new layout object 'Layout #1'
layout1 = CreateLayout(name='Layout #1')
layout1.AssignView(0, renderView1)
layout1.SetSize(2427, 1321)

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
pointVolumeInterpolator1.Source.Origin = [-70.560394, -148.83055, -0.47299883]
pointVolumeInterpolator1.Source.Scale = [0.9639206, 1.0030518, 0.9513724]

# create a new 'Threshold'
threshold1 = Threshold(registrationName='Threshold1', Input=tde_snapshot00000h5)
threshold1.Scalars = ['POINTS', '']
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
rhoLUT.RGBPoints = [1.672478333603067e-08, 0.001462, 0.000466, 0.013866, 8.318851796894932e-08, 0.002267, 0.00127, 0.01857, 1.4963530621363929e-07, 0.003299, 0.002249, 0.024239, 2.1609904084655797e-07, 0.004547, 0.003392, 0.030909, 2.825458290912479e-07, 0.006006, 0.004692, 0.038558, 3.490095637241666e-07, 0.007676, 0.006136, 0.046836, 4.1545635196885655e-07, 0.009561, 0.007713, 0.055143, 4.819200866017752e-07, 0.011663, 0.009417, 0.06346, 5.483838212346939e-07, 0.013995, 0.011225, 0.071862, 6.148306094793838e-07, 0.016561, 0.013136, 0.080282, 6.812943441123026e-07, 0.019373, 0.015133, 0.088767, 7.477411323569925e-07, 0.022447, 0.017199, 0.097327, 8.14204866989911e-07, 0.025793, 0.019331, 0.10593, 8.806516552346011e-07, 0.029432, 0.021503, 0.114621, 9.471153898675196e-07, 0.033385, 0.023702, 0.123397, 1.0135791245004385e-06, 0.037668, 0.025921, 0.132232, 1.0800259127451283e-06, 0.042253, 0.028139, 0.141141, 1.146489647378047e-06, 0.046915, 0.030324, 0.150164, 1.212936435622737e-06, 0.051644, 0.032474, 0.159254, 1.2794001702556557e-06, 0.056449, 0.034569, 0.168414, 1.3458469585003457e-06, 0.06134, 0.03659, 0.177642, 1.4123106931332643e-06, 0.066331, 0.038504, 0.186962, 1.478774427766183e-06, 0.071429, 0.040294, 0.196354, 1.5452212160108729e-06, 0.076637, 0.041905, 0.205799, 1.6116849506437914e-06, 0.081962, 0.043328, 0.215289, 1.6781317388884817e-06, 0.087411, 0.044556, 0.224813, 1.7445954735214002e-06, 0.09299, 0.045583, 0.234358, 1.8110422617660903e-06, 0.098702, 0.046402, 0.243904, 1.8775059963990086e-06, 0.104551, 0.047008, 0.25343, 1.943952784643699e-06, 0.110536, 0.047399, 0.262912, 2.0104165192766174e-06, 0.116656, 0.047574, 0.272321, 2.0768802539095357e-06, 0.122908, 0.047536, 0.281624, 2.143327042154226e-06, 0.129285, 0.047293, 0.290788, 2.2097907767871448e-06, 0.135778, 0.046856, 0.299776, 2.276237565031835e-06, 0.142378, 0.046242, 0.308553, 2.3427012996647533e-06, 0.149073, 0.045468, 0.317085, 2.409148087909443e-06, 0.15585, 0.044559, 0.325338, 2.475611822542362e-06, 0.162689, 0.043554, 0.333277, 2.5420755571752807e-06, 0.169575, 0.042489, 0.340874, 2.6085223454199705e-06, 0.176493, 0.041402, 0.348111, 2.6749860800528893e-06, 0.183429, 0.040329, 0.354971, 2.7414328682975795e-06, 0.190367, 0.039309, 0.361447, 2.807896602930498e-06, 0.197297, 0.0384, 0.367535, 2.8743433911751877e-06, 0.204209, 0.037632, 0.373238, 2.940807125808107e-06, 0.211095, 0.03703, 0.378563, 3.0072708604410252e-06, 0.217949, 0.036615, 0.383522, 3.073717648685715e-06, 0.224763, 0.036405, 0.388129, 3.140181383318634e-06, 0.231538, 0.036405, 0.3924, 3.206628171563324e-06, 0.238273, 0.036621, 0.396353, 3.2730919061962424e-06, 0.244967, 0.037055, 0.400007, 3.3395386944409327e-06, 0.25162, 0.037705, 0.403378, 3.4060024290738514e-06, 0.258234, 0.038571, 0.406485, 3.4724661637067698e-06, 0.26481, 0.039647, 0.409345, 3.5389129519514596e-06, 0.271347, 0.040922, 0.411976, 3.6053766865843784e-06, 0.27785, 0.042353, 0.414392, 3.671823474829068e-06, 0.284321, 0.043933, 0.416608, 3.738287209461987e-06, 0.290763, 0.045644, 0.418637, 3.8047339977066776e-06, 0.297178, 0.04747, 0.420491, 3.871197732339595e-06, 0.303568, 0.049396, 0.422182, 3.937661466972514e-06, 0.309935, 0.051407, 0.423721, 4.004108255217204e-06, 0.316282, 0.05349, 0.425116, 4.0705719898501225e-06, 0.32261, 0.055634, 0.426377, 4.137018778094813e-06, 0.328921, 0.057827, 0.427511, 4.203482512727732e-06, 0.335217, 0.06006, 0.428524, 4.269929300972421e-06, 0.3415, 0.062325, 0.429425, 4.3363930356053405e-06, 0.347771, 0.064616, 0.430217, 4.402856770238259e-06, 0.354032, 0.066925, 0.430906, 4.469303558482949e-06, 0.360284, 0.069247, 0.431497, 4.535767293115867e-06, 0.366529, 0.071579, 0.431994, 4.602214081360558e-06, 0.372768, 0.073915, 0.4324, 4.668677815993476e-06, 0.379001, 0.076253, 0.432719, 4.735124604238166e-06, 0.385228, 0.078591, 0.432955, 4.801588338871085e-06, 0.391453, 0.080927, 0.433109, 4.868052073504003e-06, 0.397674, 0.083257, 0.433183, 4.934498861748693e-06, 0.403894, 0.08558, 0.433179, 5.0009625963816115e-06, 0.410113, 0.087896, 0.433098, 5.067409384626302e-06, 0.416331, 0.090203, 0.432943, 5.1338731192592206e-06, 0.422549, 0.092501, 0.432714, 5.20031990750391e-06, 0.428768, 0.09479, 0.432412, 5.2667836421368296e-06, 0.434987, 0.097069, 0.432039, 5.3332304303815186e-06, 0.441207, 0.099338, 0.431594, 5.399694165014438e-06, 0.447428, 0.101597, 0.43108, 5.466157899647356e-06, 0.453651, 0.103848, 0.430498, 5.532604687892047e-06, 0.459875, 0.106089, 0.429846, 5.599068422524965e-06, 0.4661, 0.108322, 0.429125, 5.665515210769655e-06, 0.472328, 0.110547, 0.428334, 5.731978945402574e-06, 0.478558, 0.112764, 0.427475, 5.798425733647263e-06, 0.484789, 0.114974, 0.426548, 5.864889468280183e-06, 0.491022, 0.117179, 0.425552, 5.9313532029131015e-06, 0.497257, 0.119379, 0.424488, 5.997799991157791e-06, 0.503493, 0.121575, 0.423356, 6.06426372579071e-06, 0.50973, 0.123769, 0.422156, 6.1307105140353995e-06, 0.515967, 0.12596, 0.420887, 6.197174248668318e-06, 0.522206, 0.12815, 0.419549, 6.263621036913008e-06, 0.528444, 0.130341, 0.418142, 6.330084771545928e-06, 0.534683, 0.132534, 0.416667, 6.396548506178846e-06, 0.54092, 0.134729, 0.415123, 6.462995294423536e-06, 0.547157, 0.136929, 0.413511, 6.529459029056454e-06, 0.553392, 0.139134, 0.411829, 6.595905817301144e-06, 0.559624, 0.141346, 0.410078, 6.662369551934062e-06, 0.565854, 0.143567, 0.408258, 6.728816340178753e-06, 0.572081, 0.145797, 0.406369, 6.795280074811672e-06, 0.578304, 0.148039, 0.404411, 6.8617438094445905e-06, 0.584521, 0.150294, 0.402385, 6.92819059768928e-06, 0.590734, 0.152563, 0.40029, 6.994654332322199e-06, 0.59694, 0.154848, 0.398125, 7.0611011205668885e-06, 0.603139, 0.157151, 0.395891, 7.127564855199807e-06, 0.60933, 0.159474, 0.393589, 7.1940116434444975e-06, 0.615513, 0.161817, 0.391219, 7.260475378077417e-06, 0.621685, 0.164184, 0.388781, 7.326939112710335e-06, 0.627847, 0.166575, 0.386276, 7.393385900955025e-06, 0.633998, 0.168992, 0.383704, 7.459849635587943e-06, 0.640135, 0.171438, 0.381065, 7.526296423832633e-06, 0.64626, 0.173914, 0.378359, 7.5927601584655506e-06, 0.652369, 0.176421, 0.375586, 7.659206946710242e-06, 0.658463, 0.178962, 0.372748, 7.72567068134316e-06, 0.66454, 0.181539, 0.369846, 7.792134415976079e-06, 0.670599, 0.184153, 0.366879, 7.85858120422077e-06, 0.676638, 0.186807, 0.363849, 7.925044938853687e-06, 0.682656, 0.189501, 0.360757, 7.991491727098378e-06, 0.688653, 0.192239, 0.357603, 8.057955461731297e-06, 0.694627, 0.195021, 0.354388, 8.124402249975986e-06, 0.700576, 0.197851, 0.351113, 8.190865984608905e-06, 0.7065, 0.200728, 0.347777, 8.257329719241824e-06, 0.712396, 0.203656, 0.344383, 8.323776507486515e-06, 0.718264, 0.206636, 0.340931, 8.390240242119434e-06, 0.724103, 0.20967, 0.337424, 8.456687030364123e-06, 0.729909, 0.212759, 0.333861, 8.52315076499704e-06, 0.735683, 0.215906, 0.330245, 8.589597553241731e-06, 0.741423, 0.219112, 0.326576, 8.65606128787465e-06, 0.747127, 0.222378, 0.322856, 8.72250807611934e-06, 0.752794, 0.225706, 0.319085, 8.788971810752258e-06, 0.758422, 0.229097, 0.315266, 8.855435545385176e-06, 0.76401, 0.232554, 0.311399, 8.921882333629867e-06, 0.769556, 0.236077, 0.307485, 8.988346068262786e-06, 0.775059, 0.239667, 0.303526, 9.054792856507475e-06, 0.780517, 0.243327, 0.299523, 9.121256591140396e-06, 0.785929, 0.247056, 0.295477, 9.187703379385085e-06, 0.791293, 0.250856, 0.29139, 9.254167114018002e-06, 0.796607, 0.254728, 0.287264, 9.320630848650921e-06, 0.801871, 0.258674, 0.283099, 9.387077636895612e-06, 0.807082, 0.262692, 0.278898, 9.45354137152853e-06, 0.812239, 0.266786, 0.274661, 9.519988159773219e-06, 0.817341, 0.270954, 0.27039, 9.58645189440614e-06, 0.822386, 0.275197, 0.266085, 9.652898682650828e-06, 0.827372, 0.279517, 0.26175, 9.719362417283748e-06, 0.832299, 0.283913, 0.257383, 9.785826151916665e-06, 0.837165, 0.288385, 0.252988, 9.852272940161356e-06, 0.841969, 0.292933, 0.248564, 9.918736674794275e-06, 0.846709, 0.297559, 0.244113, 9.985183463038964e-06, 0.851384, 0.30226, 0.239636, 1.0051647197671885e-05, 0.855992, 0.307038, 0.235133, 1.0118093985916574e-05, 0.860533, 0.311892, 0.230606, 1.0184557720549491e-05, 0.865006, 0.316822, 0.226055, 1.025102145518241e-05, 0.869409, 0.321827, 0.221482, 1.0317468243427101e-05, 0.873741, 0.326906, 0.216886, 1.0383931978060019e-05, 0.878001, 0.33206, 0.212268, 1.0450378766304708e-05, 0.882188, 0.337287, 0.207628, 1.0516842500937628e-05, 0.886302, 0.342586, 0.202968, 1.0583289289182317e-05, 0.890341, 0.347957, 0.198286, 1.0649753023815237e-05, 0.894305, 0.353399, 0.193584, 1.0716216758448154e-05, 0.898192, 0.358911, 0.18886, 1.0782663546692845e-05, 0.902003, 0.364492, 0.184116, 1.0849127281325764e-05, 0.905735, 0.37014, 0.17935, 1.0915574069570453e-05, 0.90939, 0.375856, 0.174563, 1.0982037804203374e-05, 0.912966, 0.381636, 0.169755, 1.1048484592448063e-05, 0.916462, 0.387481, 0.164924, 1.111494832708098e-05, 0.919879, 0.393389, 0.16007, 1.11814120617139e-05, 0.923215, 0.399359, 0.155193, 1.124785884995859e-05, 0.92647, 0.405389, 0.150292, 1.1314322584591508e-05, 0.929644, 0.411479, 0.145367, 1.1380769372836198e-05, 0.932737, 0.417627, 0.140417, 1.1447233107469118e-05, 0.935747, 0.423831, 0.13544, 1.1513679895713807e-05, 0.938675, 0.430091, 0.130438, 1.1580143630346726e-05, 0.941521, 0.436405, 0.125409, 1.1646607364979643e-05, 0.944285, 0.442772, 0.120354, 1.1713054153224336e-05, 0.946965, 0.449191, 0.115272, 1.1779517887857253e-05, 0.949562, 0.45566, 0.110164, 1.1845964676101942e-05, 0.952075, 0.462178, 0.105031, 1.1912428410734861e-05, 0.954506, 0.468744, 0.099874, 1.1978875198979552e-05, 0.956852, 0.475356, 0.094695, 1.204533893361247e-05, 0.959114, 0.482014, 0.089499, 1.2111785721857162e-05, 0.961293, 0.488716, 0.084289, 1.217824945649008e-05, 0.963387, 0.495462, 0.079073, 1.2244713191122998e-05, 0.965397, 0.502249, 0.073859, 1.2311159979367687e-05, 0.967322, 0.509078, 0.068659, 1.2377623714000605e-05, 0.969163, 0.515946, 0.063488, 1.2444070502245296e-05, 0.970919, 0.522853, 0.058367, 1.2510534236878215e-05, 0.97259, 0.529798, 0.053324, 1.2576981025122904e-05, 0.974176, 0.53678, 0.048392, 1.2643444759755825e-05, 0.975677, 0.543798, 0.043618, 1.2709908494388742e-05, 0.977092, 0.55085, 0.03905, 1.2776355282633431e-05, 0.978422, 0.557937, 0.034931, 1.284281901726635e-05, 0.979666, 0.565057, 0.031409, 1.2909265805511041e-05, 0.980824, 0.572209, 0.028508, 1.2975729540143958e-05, 0.981895, 0.579392, 0.02625, 1.3042176328388647e-05, 0.982881, 0.586606, 0.024661, 1.3108640063021568e-05, 0.983779, 0.593849, 0.02377, 1.3175103797654488e-05, 0.984591, 0.601122, 0.023606, 1.3241550585899176e-05, 0.985315, 0.608422, 0.024202, 1.3308014320532094e-05, 0.985952, 0.61575, 0.025592, 1.3374461108776785e-05, 0.986502, 0.623105, 0.027814, 1.3440924843409704e-05, 0.986964, 0.630485, 0.030908, 1.3507371631654393e-05, 0.987337, 0.63789, 0.034916, 1.3573835366287314e-05, 0.987622, 0.64532, 0.039886, 1.3640299100920231e-05, 0.987819, 0.652773, 0.045581, 1.370674588916492e-05, 0.987926, 0.66025, 0.05175, 1.377320962379784e-05, 0.987945, 0.667748, 0.058329, 1.383965641204253e-05, 0.987874, 0.675267, 0.065257, 1.3906120146675448e-05, 0.987714, 0.682807, 0.072489, 1.3972566934920138e-05, 0.987464, 0.690366, 0.07999, 1.4039030669553057e-05, 0.987124, 0.697944, 0.087731, 1.4105494404185977e-05, 0.986694, 0.70554, 0.095694, 1.4171941192430666e-05, 0.986175, 0.713153, 0.103863, 1.4238404927063583e-05, 0.985566, 0.720782, 0.112229, 1.4304851715308275e-05, 0.984865, 0.728427, 0.120785, 1.4371315449941193e-05, 0.984075, 0.736087, 0.129527, 1.4437762238185882e-05, 0.983196, 0.743758, 0.138453, 1.4504225972818803e-05, 0.982228, 0.751442, 0.147565, 1.457068970745172e-05, 0.981173, 0.759135, 0.156863, 1.463713649569641e-05, 0.980032, 0.766837, 0.166353, 1.4703600230329328e-05, 0.978806, 0.774545, 0.176037, 1.4770047018574019e-05, 0.977497, 0.782258, 0.185923, 1.4836510753206938e-05, 0.976108, 0.789974, 0.196018, 1.4902957541451627e-05, 0.974638, 0.797692, 0.206332, 1.4969421276084546e-05, 0.973088, 0.805409, 0.216877, 1.5035885010717466e-05, 0.971468, 0.813122, 0.227658, 1.5102331798962155e-05, 0.969783, 0.820825, 0.238686, 1.5168795533595072e-05, 0.968041, 0.828515, 0.249972, 1.5235242321839764e-05, 0.966243, 0.836191, 0.261534, 1.530170605647268e-05, 0.964394, 0.843848, 0.273391, 1.536815284471737e-05, 0.962517, 0.851476, 0.285546, 1.543461657935029e-05, 0.960626, 0.859069, 0.29801, 1.5501063367594984e-05, 0.95872, 0.866624, 0.31082, 1.55675271022279e-05, 0.956834, 0.874129, 0.323974, 1.5633990836860812e-05, 0.954997, 0.881569, 0.337475, 1.5700437625105508e-05, 0.953215, 0.888942, 0.351369, 1.5766901359738426e-05, 0.951546, 0.896226, 0.365627, 1.5833348147983115e-05, 0.950018, 0.903409, 0.380271, 1.589981188261604e-05, 0.948683, 0.910473, 0.395289, 1.5966258670860725e-05, 0.947594, 0.917399, 0.410665, 1.6032722405493642e-05, 0.946809, 0.924168, 0.426373, 1.6099186140126563e-05, 0.946392, 0.930761, 0.442367, 1.6165632928371255e-05, 0.946403, 0.937159, 0.458592, 1.623209666300417e-05, 0.946903, 0.943348, 0.47497, 1.6298543451248862e-05, 0.947937, 0.949318, 0.491426, 1.636500718588178e-05, 0.949545, 0.955063, 0.50786, 1.643145397412647e-05, 0.95174, 0.960587, 0.524203, 1.6497917708759386e-05, 0.954529, 0.965896, 0.540361, 1.656438144339231e-05, 0.957896, 0.971003, 0.556275, 1.6630828231637e-05, 0.961812, 0.975924, 0.571925, 1.6697291966269916e-05, 0.966249, 0.980678, 0.587206, 1.6763738754514602e-05, 0.971162, 0.985282, 0.602154, 1.6830202489147526e-05, 0.976511, 0.989753, 0.61676, 1.6896649277392215e-05, 0.982257, 0.994109, 0.631017, 1.6963113012025133e-05, 0.988362, 0.998364, 0.644924]
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
rhoPWF.Points = [1.672478333603067e-08, 0.0, 0.5, 0.0, 6.910996799762796e-07, 0.0, 0.5, 0.0, 1.5830147864181151e-06, 0.03804348036646843, 0.5, 0.0, 2.4096677478343647e-06, 0.032608695328235626, 0.5, 0.0, 7.184273141293479e-06, 0.03804348036646843, 0.5, 0.0, 1.2155472573106564e-05, 0.07608696073293686, 0.5, 0.0, 1.4711999077516227e-05, 0.2554347813129425, 0.5, 0.0, 1.6278774332079565e-05, 1.0, 0.5, 0.0, 1.6963113012025133e-05, 1.0, 0.5, 0.0]
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
threshold1Display.Representation = 'Surface'
threshold1Display.ColorArrayName = ['POINTS', 'rho']
threshold1Display.LookupTable = rhoLUT
threshold1Display.Opacity = 0.14
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
threshold1Display.ScalarOpacityFunction = rhoPWF
threshold1Display.ScalarOpacityUnitDistance = 0.025306868557695913
threshold1Display.OpacityArrayName = ['POINTS', 'rho']
threshold1Display.SelectInputVectors = ['POINTS', '']
threshold1Display.WriteLog = ''

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
threshold1Display.ScaleTransferFunction.Points = [9.366737430127614e-08, 0.0, 0.5, 0.0, 1.9999995402031345e-06, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
threshold1Display.OpacityTransferFunction.Points = [9.366737430127614e-08, 0.0, 0.5, 0.0, 1.9999995402031345e-06, 1.0, 0.5, 0.0]

# show data from sphere1
sphere1Display = Show(sphere1, renderView1, 'GeometryRepresentation')

# trace defaults for the display properties.
sphere1Display.Representation = 'Surface'
sphere1Display.AmbientColor = [0.0, 0.0, 0.0]
sphere1Display.ColorArrayName = [None, '']
sphere1Display.DiffuseColor = [0.0, 0.0, 0.0]
sphere1Display.Opacity = 0.3
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

# show color legend
threshold1Display.SetScalarBarVisibility(renderView1, True)

# ----------------------------------------------------------------
# setup color maps and opacity mapes used in the visualization
# note: the Get..() functions create a new object, if needed
# ----------------------------------------------------------------

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

# create extractor
pNG1 = CreateExtractor('PNG', renderView1, registrationName='PNG1')
# trace defaults for the extractor.
pNG1.Trigger = 'TimeStep'

# init the 'TimeStep' selected for 'Trigger'
pNG1.Trigger.UseStartTimeStep = 1
pNG1.Trigger.UseEndTimeStep = 1
pNG1.Trigger.EndTimeStep = 300

# init the 'PNG' selected for 'Writer'
pNG1.Writer.FileName = 'RenderView1_{timestep:06d}{camera}.png'
pNG1.Writer.ImageResolution = [2427, 1321]
pNG1.Writer.Format = 'PNG'

# ----------------------------------------------------------------
# restore active source
SetActiveSource(threshold1)
# ----------------------------------------------------------------


if __name__ == '__main__':
    # generate extracts
    SaveExtracts(ExtractsOutputDirectory='tde_3d')
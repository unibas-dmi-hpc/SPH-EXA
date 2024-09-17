# Python script for python animation track in ParaView
# Fitting the origin and scale of filter PointVolumeInterpolator1 to the dataset in every frame
# It doesn't work perfectly because the coordinates of dataset (data_source.GetDataInformation().GetBounds()) has a bit of delay. It doesn't correspond to the frames exactly.
# To ensure a complete visualization (i.e. boundary of PointVolumeInterpolator1 fits to boundary of dataset), either extend the origin and scale of interpolator to cover a bit more than the dataset
# Or manually insert keyframes (very slow, but works)
def start_cue(self): pass

def tick(self):
  from paraview.simple import FindSource
  t = self.GetAnimationTime()
  data_source = FindSource('tde_snapshot00000.h5')
  filter_source = FindSource('PointVolumeInterpolator1')
  bds = data_source.GetDataInformation().GetBounds()
  filter_source.Source.Origin = [bds[0], bds[2], bds[4]]
  filter_source.Source.Scale = [bds[1] - bds[0], bds[3] - bds[2], bds[5] - bds[4]]
  filter_source.UpdatePipeline()
  print(filter_source.Source.Origin)
  print(bds)

def end_cue(self): pass
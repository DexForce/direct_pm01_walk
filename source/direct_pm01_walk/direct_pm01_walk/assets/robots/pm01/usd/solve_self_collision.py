from pxr import Usd, UsdPhysics, Sdf
# pip install usd-core


stage = Usd.Stage.Open("pm01.usd")

pairs = [
    ("/pm01/link_ankle_pitch_r", "/pm01/link_ankle_roll_r"),
    ("/pm01/link_ankle_pitch_r", "/pm01/link_knee_pitch_r"),
    ("/pm01/link_ankle_roll_r",  "/pm01/link_knee_pitch_r"),
    ("/pm01/link_ankle_pitch_l", "/pm01/link_ankle_roll_l"),
    ("/pm01/link_ankle_pitch_l", "/pm01/link_knee_pitch_l"),
    ("/pm01/link_ankle_roll_l",  "/pm01/link_knee_pitch_l"),
]

def add_pair(a, b):
    primA = stage.GetPrimAtPath(a)
    apiA  = UsdPhysics.FilteredPairsAPI.Apply(primA)

    relA = apiA.GetFilteredPairsRel()
    if not relA:
        relA = apiA.CreateFilteredPairsRel()

    relA.AddTarget(Sdf.Path(b))

for a, b in pairs:
    add_pair(a, b)
    add_pair(b, a)

stage.GetRootLayer().Save()

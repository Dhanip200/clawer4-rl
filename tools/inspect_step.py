"""Print the STEP assembly tree with world-frame bounding boxes (mm)."""
import sys
from OCP.STEPCAFControl import STEPCAFControl_Reader
from OCP.TDocStd import TDocStd_Document
from OCP.TCollection import TCollection_ExtendedString
from OCP.XCAFDoc import XCAFDoc_DocumentTool
from OCP.TDF import TDF_Label
from OCP.collections import Sequence_TDF_Label as TDF_LabelSequence
from OCP.TDataStd import TDataStd_Name
from OCP.Bnd import Bnd_Box
from OCP.BRepBndLib import BRepBndLib
from OCP.TopLoc import TopLoc_Location

path = sys.argv[1]
max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 3
doc = TDocStd_Document(TCollection_ExtendedString("doc"))
r = STEPCAFControl_Reader(); r.SetNameMode(True)
r.ReadFile(path); r.Transfer(doc)
st = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())

def name(lab):
    a = TDataStd_Name()
    return a.Get().ToExtString() if lab.FindAttribute(TDataStd_Name.GetID_s(), a) else "?"

def walk(lab, loc, depth):
    ref = TDF_Label()
    if st.IsReference_s(lab):
        st.GetReferredShape_s(lab, ref)
        loc = loc.Multiplied(st.GetLocation_s(lab))
    else:
        ref = lab
    shape = st.GetShape_s(ref).Moved(loc)
    b = Bnd_Box(); BRepBndLib.Add_s(shape, b)
    p0, p1 = b.CornerMin(), b.CornerMax(); x0, y0, z0, x1, y1, z1 = p0.X(), p0.Y(), p0.Z(), p1.X(), p1.Y(), p1.Z()
    t = loc.Transformation().TranslationPart()
    print(f"{'  '*depth}{name(lab)} [{name(ref)}] org=({t.X():.1f},{t.Y():.1f},{t.Z():.1f}) "
          f"bb=({x0:.1f},{y0:.1f},{z0:.1f})-({x1:.1f},{y1:.1f},{z1:.1f})")
    if depth < max_depth:
        kids = TDF_LabelSequence(); st.GetComponents_s(ref, kids)
        for i in range(1, kids.Length() + 1):
            walk(kids.Value(i), loc, depth + 1)

roots = TDF_LabelSequence(); st.GetFreeShapes(roots)
for i in range(1, roots.Length() + 1):
    walk(roots.Value(i), TopLoc_Location(), 0)

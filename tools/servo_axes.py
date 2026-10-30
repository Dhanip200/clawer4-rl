"""For leg 1, print each servo's sub-parts with world bbox + rotation, to locate output shafts."""
import sys
sys.argv = [sys.argv[0], "cad/CLAWER-4 based on ST3020 Servo.step", "0"]
exec(open("tools/inspect_step.py").read().split("roots = ")[0])
def dump(lab, loc, depth, maxd):
    ref = TDF_Label()
    if st.IsReference_s(lab):
        st.GetReferredShape_s(lab, ref); loc = loc.Multiplied(st.GetLocation_s(lab))
    else: ref = lab
    shape = st.GetShape_s(ref).Moved(loc)
    b = Bnd_Box(); BRepBndLib.Add_s(shape, b); p0, p1 = b.CornerMin(), b.CornerMax()
    T = loc.Transformation()
    R = [[T.Value(i, j) for j in (1, 2, 3)] for i in (1, 2, 3)]
    print(f"{'  '*depth}{name(lab)} bb=({p0.X():.1f},{p0.Y():.1f},{p0.Z():.1f})-({p1.X():.1f},{p1.Y():.1f},{p1.Z():.1f}) zcol=({R[0][2]:.2f},{R[1][2]:.2f},{R[2][2]:.2f})")
    if depth < maxd:
        kids = TDF_LabelSequence(); st.GetComponents_s(ref, kids)
        for i in range(1, kids.Length() + 1): dump(kids.Value(i), loc, depth + 1, maxd)
roots = TDF_LabelSequence(); st.GetFreeShapes(roots)
top = roots.Value(1); topref = top
kids = TDF_LabelSequence(); st.GetComponents_s(top, kids)
leg = kids.Value(2)  # CLAWER LEG v3:1
ref = TDF_Label(); st.GetReferredShape_s(leg, ref)
lk = TDF_LabelSequence(); st.GetComponents_s(ref, lk)
for i in range(1, lk.Length() + 1):
    if "Servo" in name(lk.Value(i)):
        dump(lk.Value(i), st.GetLocation_s(leg), 0, 2)

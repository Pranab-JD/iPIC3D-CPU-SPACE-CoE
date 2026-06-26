pip install --user vtk

file="/scratch/project_465002078/pranab/RMR/sigma_1/domain_55_110_55/Bz_0.25/CS2/CS2_cycle_17500.vtk"

python3 - <<'PY' "$file"
import sys
import os
import re

fname = sys.argv[1]
ext = os.path.splitext(fname)[1].lower()

if ext == ".vtk":
    from vtkmodules.vtkIOLegacy import vtkDataSetReader
    reader = vtkDataSetReader()
    reader.SetFileName(fname)
    reader.ReadAllScalarsOn()
    reader.ReadAllVectorsOn()
    reader.ReadAllFieldsOn()
    reader.Update()

    data = (reader.GetStructuredPointsOutput()
         or reader.GetStructuredGridOutput()
         or reader.GetRectilinearGridOutput()
         or reader.GetUnstructuredGridOutput()
         or reader.GetPolyDataOutput())
else:
    raise RuntimeError(f"Unsupported file extension: {ext}")

if data is None:
    raise RuntimeError(f"Could not read dataset from {fname}")

print("=============== FILE INFO ===============")
print("File:", fname)
print("Class:", data.GetClassName())

bounds = data.GetBounds()
print("\n=============== BOUNDS ===============")
print(f"X: [{bounds[0]}, {bounds[1]}]")
print(f"Y: [{bounds[2]}, {bounds[3]}]")
print(f"Z: [{bounds[4]}, {bounds[5]}]")

dims = [0, 0, 0]
if hasattr(data, "GetDimensions"):
    data.GetDimensions(dims)
    nx, ny, nz = dims
    print(f"Cell dimensions : {max(nx-1,0)} x {max(ny-1,0)} x {max(nz-1,0)}")



print("\n=============== DATA ===============")
m = re.search(r'cycle_(\d+)', os.path.basename(fname))
if m:
    print("\nCycle (from filename):", m.group(1))

pd = data.GetPointData()

if pd is None or pd.GetNumberOfArrays() == 0:
    print("No PointData arrays")
else:
    names = []
    for i in range(pd.GetNumberOfArrays()):
        arr = pd.GetAbstractArray(i)
        name = pd.GetArrayName(i) or (arr.GetName() if arr is not None else None) or f"<unnamed_{i}>"
        names.append(name)

    print("Names:", names)
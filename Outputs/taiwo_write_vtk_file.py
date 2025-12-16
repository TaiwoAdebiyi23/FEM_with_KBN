import numpy as np
from pathlib import Path
from typing import Literal


def write_vtk_unstructured(
    connectivity: np.ndarray,
    coord: np.ndarray,
    data: np.ndarray,
    ele_type: str,
    filename: str,
    directory: Path,
    data_type: Literal["Scalar", "Vector"] = "Scalar",
    data_name: str = "Data",
) -> None:
    """
    Write an unstructured-grid legacy VTK file (ASCII), similar to
    KBN's WriteVTKFile.m.

    Parameters
    ----------
    connectivity : (Nele, NodesPerEle) int array
        Element connectivity with 1-based node numbering.
    coord : (NumNodes, dim) float array
        Nodal coordinates. dim can be 1, 2, or 3; padded to 3D for VTK.
    data : array
        Nodal data:
          - if data_type == "Scalar": shape (NumNodes,) or (NumNodes, 1)
          - if data_type == "Vector": shape (NumNodes, dim_data), dim_data<=3
    ele_type : str
        Element type: 'Q4', 'Q8', 'T3', 'B8', 'TET4', ...
    filename : str
        Base filename (with or without .vtk).
    directory : Path
        Directory where the .vtk file will be written.
    data_type : {"Scalar", "Vector"}
        Whether to write the data as a scalar or a vector field.
    data_name : str
        Name of the data field in the VTK file.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    if not filename.lower().endswith(".vtk"):
        filename = filename + ".vtk"
    full_path = directory / filename

    coord = np.asarray(coord, dtype=float)
    connectivity = np.asarray(connectivity, dtype=int)
    data = np.asarray(data, dtype=float)

    num_nodes, dim = coord.shape
    nele, nodes_per_ele = connectivity.shape

    if dim < 1 or dim > 3:
        raise ValueError("coord must have 1, 2, or 3 columns (x, y, [z]).")

    # Map element type to VTK cell type ID
    ele_type = ele_type.upper()
    cell_type_map = {
        "B8": 12,     # VTK_HEXAHEDRON
        "Q4": 9,      # VTK_QUAD
        "Q8": 23,     # VTK_QUADRATIC_QUAD (4 corners + 4 mids)
        "T3": 5,      # VTK_TRIANGLE
        "TET4": 10,   # VTK_TETRA
    }
    if ele_type not in cell_type_map:
        raise ValueError(
            f"Element type '{ele_type}' not supported. "
            f"Supported: {list(cell_type_map.keys())}"
        )
    cell_id = cell_type_map[ele_type]

    with full_path.open("w", encoding="utf-8") as f:
        # Header
        f.write("# vtk DataFile Version 2.0\n")
        f.write("Written using Python VTK writer (Taiwo)\n")
        f.write("ASCII\n\n")

        # Dataset
        f.write("DATASET UNSTRUCTURED_GRID\n")

        # POINTS
        f.write(f"POINTS {num_nodes} float\n")
        xyz = np.zeros((num_nodes, 3), dtype=float)
        xyz[:, :dim] = coord
        for p in xyz:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
        f.write("\n")

        # CELLS (connectivity)
        total_ints = nele * (nodes_per_ele + 1)
        f.write(f"CELLS {nele} {total_ints}\n")
        conn0 = connectivity - 1
        for row in conn0:
            row_str = " ".join(str(int(n)) for n in row)
            f.write(f"{nodes_per_ele} {row_str}\n")
        f.write("\n")

        # CELL_TYPES
        f.write(f"CELL_TYPES {nele}\n")
        for _ in range(nele):
            f.write(f"{cell_id}\n")
        f.write("\n")

        # POINT_DATA
        if data_type.lower() == "scalar":
            dat = data.reshape(-1)
            if dat.size != num_nodes:
                raise ValueError(
                    f"Scalar data must have NumNodes entries, got {dat.size}."
                )
            f.write(f"POINT_DATA {num_nodes}\n")
            f.write(f"SCALARS {data_name} float\n")
            f.write("LOOKUP_TABLE default\n")
            for val in dat:
                f.write(f"{val:.8f}\n")
            f.write("\n")

        elif data_type.lower() == "vector":
            if data.ndim != 2 or data.shape[0] != num_nodes:
                raise ValueError(
                    "Vector data must have shape (NumNodes, dim_data)."
                )
            dim_data = data.shape[1]
            if dim_data < 1 or dim_data > 3:
                raise ValueError("Vector data must have 1–3 components per node.")

            vec = np.zeros((num_nodes, 3), dtype=float)
            vec[:, :dim_data] = data

            f.write(f"POINT_DATA {num_nodes}\n")
            f.write(f"VECTORS {data_name} float\n")
            for row in vec:
                f.write(f"{row[0]:.8f} {row[1]:.8f} {row[2]:.8f}\n")
            f.write("\n")
        else:
            raise ValueError("data_type must be 'Scalar' or 'Vector'.")

    print(f"VTK file written to: {full_path}")

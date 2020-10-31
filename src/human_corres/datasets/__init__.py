try:
  from .faust_mesh import FaustMesh, FaustTestMesh
  from .shrec19_mesh import Shrec19Mesh
  from .surreal import SurrealFEPts
  #from .dgfsurreal import DGFSurrealFEPts
  from .surreal_mesh import SurrealMesh
  #from .surreal_img import SurrealFEDepthImgs
  from .faust import FaustFEPts, FaustTestFEPts
  #from .faust_img import FaustFEDepthImgs
  from .shrec19 import Shrec19FEPts
  from .tosca_mesh import ToscaMesh
  from .tosca import ToscaFEPts
  from .smal import SMALFEPts
except Exception as e:
  print('error init human_corres.datasets:', e)

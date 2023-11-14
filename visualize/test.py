from direct.showbase.ShowBase import ShowBase
from panda3d.core import Geom, GeomNode, GeomVertexFormat, GeomVertexData, GeomTriangles, GeomVertexWriter, NodePath
from panda3d.core import Point3, DirectionalLight, AmbientLight

class HandModel(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)
        self.disableMouse()  # Disable the default mouse camera control
        
        self.setup_scene()

        self.hand_model = self.create_hand_model()
        self.hand_model.reparentTo(self.render)

        self.setup_lighting()

    def setup_scene(self):
        # Set background color
        self.setBackgroundColor(0.8, 0.8, 0.8, 1)

        # Create a camera
        self.camera.setPos(0, -10, 0)  # Adjust camera position
        self.camera.lookAt(0, 0, 0)  # Point camera to the origin

    def setup_lighting(self):
        # Create directional light
        d_light = DirectionalLight('d_light')
        d_light.setColor((1, 1, 1, 1))
        d_light_np = self.render.attachNewNode(d_light)
        d_light_np.setHpr(45, -45, 0)  # Adjust light direction
        self.render.setLight(d_light_np)

        # Create ambient light
        a_light = AmbientLight('a_light')
        a_light.setColor((0.2, 0.2, 0.2, 1))
        a_light_np = self.render.attachNewNode(a_light)
        self.render.setLight(a_light_np)

    def create_hand_model(self):
        # Create a NodePath to hold the hand model
        hand_np = NodePath("HandModel")

        # Define hand vertices (for a basic hand model)
        vertices = [
            Point3(0.0, 0.0, 0.0),  # Wrist point (base)
            Point3(0.1, 0.2, 0.0),  # Example: Finger 1
            Point3(-0.1, 0.2, 0.0),  # Example: Finger 2
            # Add more points as needed for fingers and palm
        ]

        # Define hand geometry
        format = GeomVertexFormat.getV3()
        vdata = GeomVertexData("handData", format, Geom.UHStatic)
        vertex_writer = GeomVertexWriter(vdata, "vertex")

        for vertex in vertices:
            vertex_writer.addData3f(vertex)

        # Define hand triangles (connecting vertices)
        tris = GeomTriangles(Geom.UHStatic)
        # Define triangles to create the hand shape (based on vertex indices)
        # Add triangles to the tris object

        # Create a Geom and attach the vertices and triangles
        geom = Geom(vdata)
        geom.addPrimitive(tris)

        # Create a GeomNode to hold the hand geometry
        hand_geom_node = GeomNode("hand")
        hand_geom_node.addGeom(geom)

        # Attach the GeomNode to the hand NodePath
        hand_np.attachNewNode(hand_geom_node)
        return hand_np

if __name__ == "__main__":
    app = HandModel()
    app.run()

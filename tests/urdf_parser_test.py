from urdf_parser_py.urdf import URDF
robot = URDF.from_xml_file('/tmp/sim_mount.urdf')
print([l.name for l in robot.links])
print([(j.name, j.type, j.parent, j.child) for j in robot.joints])
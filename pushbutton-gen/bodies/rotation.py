import cadquery as cq
s = cq.importers.importStep("3su_Body_1.STEP").val()
s = s.rotate((0, 0, 0), (1, 0, 0), -90)  
cq.exporters.export(s, "3su_Body_zup.STEP")
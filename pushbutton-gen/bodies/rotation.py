import cadquery as cq
s = cq.importers.importStep("STC pushbutton.STEP").val()
s = s.rotate((0, 0, 0), (1, 0, 0), 180)  
cq.exporters.export(s, "STC_pushbutton_zup.STEP")
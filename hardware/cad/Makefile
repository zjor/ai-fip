OPENSCAD ?= openscad
OSFLAGS  ?= --backend=manifold
BUILD    := build
PY       ?= python3

PARTS := wheel_quadrant wheel_hub wheel_bolts beam motor_flange stand pad

.PHONY: parts masses check view test clean

parts: $(addprefix $(BUILD)/,$(addsuffix .stl,$(PARTS)))

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/wheel_quadrant.stl: wheel.scad params.scad lib.scad | $(BUILD)
	$(OPENSCAD) $(OSFLAGS) -D 'part="quadrant"' -o $@ $<
$(BUILD)/wheel_hub.stl: wheel.scad params.scad lib.scad | $(BUILD)
	$(OPENSCAD) $(OSFLAGS) -D 'part="hub"' -o $@ $<
$(BUILD)/wheel_bolts.stl: wheel.scad params.scad lib.scad | $(BUILD)
	$(OPENSCAD) $(OSFLAGS) -D 'part="bolts"' -o $@ $<
$(BUILD)/beam.stl: rod.scad params.scad lib.scad | $(BUILD)
	$(OPENSCAD) $(OSFLAGS) -D 'part="beam"' -o $@ $<
$(BUILD)/motor_flange.stl: rod.scad params.scad lib.scad | $(BUILD)
	$(OPENSCAD) $(OSFLAGS) -D 'part="flange"' -o $@ $<
$(BUILD)/stand.stl: stand.scad params.scad lib.scad | $(BUILD)
	$(OPENSCAD) $(OSFLAGS) -D 'part="stand"' -o $@ $<
$(BUILD)/pad.stl: stand.scad params.scad lib.scad | $(BUILD)
	$(OPENSCAD) $(OSFLAGS) -D 'part="pad"' -o $@ $<
$(BUILD)/assembly.stl: assembly.scad wheel.scad rod.scad stand.scad params.scad lib.scad | $(BUILD)
	$(OPENSCAD) $(OSFLAGS) -o $@ $<

masses: parts
	$(PY) tools/stl_props.py --density 1.25e-3 --axis z $(BUILD)/wheel_quadrant.stl $(BUILD)/wheel_hub.stl
	$(PY) tools/stl_props.py --density 7.85e-3 --axis z $(BUILD)/wheel_bolts.stl
	$(PY) tools/stl_props.py --density 1.25e-3 --axis y $(BUILD)/beam.stl $(BUILD)/motor_flange.stl $(BUILD)/stand.stl $(BUILD)/pad.stl
	$(PY) tools/mass_model.py

# OpenSCAD exits non-zero when the intersection is empty; that is the pass case.
check: check.scad assembly.scad wheel.scad rod.scad stand.scad params.scad lib.scad | $(BUILD)
	@out=$$($(OPENSCAD) $(OSFLAGS) -o $(BUILD)/check.stl check.scad 2>&1); \
	if echo "$$out" | grep -q "top level object is empty"; then echo "check: no interference"; \
	elif [ -f $(BUILD)/check.stl ] && $(PY) tools/stl_props.py $(BUILD)/check.stl | grep -q "volume_mm3=0.0 "; then echo "check: no interference"; \
	else echo "check: INTERFERENCE"; echo "$$out"; exit 1; fi

view: $(BUILD)/assembly.stl
	@echo "open $(BUILD)/assembly.stl in a viewer, or: openscad assembly.scad"

test:
	$(PY) -m unittest discover -s tests -v

clean:
	rm -rf $(BUILD)

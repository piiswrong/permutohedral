PERMUTOHEDRAL_SRC = $(wildcard plugin/permutohedral/*.cc)
PLUGIN_OBJ += $(patsubst %.cc, build/%.o, $(PERMUTOHEDRAL_SRC))
PERMUTOHEDRAL_CUSRC = $(wildcard plugin/permutohedral/*.cu)
PLUGIN_CUOBJ += $(patsubst %.cu, build/%_gpu.o, $(PERMUTOHEDRAL_CUSRC))

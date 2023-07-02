.PHONY: all
all:help

.PHONY: build-hello # Build hello
build-hello:
	nvcc hello_world.cu -o hello_world

.PHONY: build-add # Build add
build-add:
	nvcc add.cu -o add

.PHONY: build-add-parallel # Build add_parallel
build-add-parallel:
	nvcc add_parallel.cu -o add_parallel

.PHONY: build-add-parallel-threads # Build add_parallel_threads
build-add-parallel-threads:
	nvcc add_parallel_threads.cu -o add_parallel_threads

.PHONY: format
format:
	clang-format -i *.cu

.PHONY: clean # Clean all
clean:
	rm -rf hello_world
	rm -rf add
	rm -rf add_parallel
	rm -rf add_parallel_threads

.PHONY: help # Generate list of targets with descriptions
help:
	@grep '^.PHONY: .* #' Makefile | sed 's/\.PHONY: \(.*\) # \(.*\)/\1	\2/' | expand -t20
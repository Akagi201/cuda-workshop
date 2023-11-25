use cuda_builder::CudaBuilder;

fn main() {
    CudaBuilder::new("../add_gpu")
        .copy_to("../resources/add.ptx")
        .build()
        .unwrap();
}
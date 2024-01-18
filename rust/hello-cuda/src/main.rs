use anyhow::Result;
use cudarc::driver::{LaunchAsync, LaunchConfig};

const PTX_SRC: &str = r#"
    extern "C" __global__ void hello_cuda_from_gpu(size_t num) {
        printf("GPU: Hello, CUDA!\n");
    }"#;

fn main() -> Result<()> {
    {
        // GPU
        let dev = cudarc::driver::CudaDevice::new(0)?;
        let ptx = cudarc::nvrtc::compile_ptx(PTX_SRC)?;
        dev.load_ptx(ptx, "hello_cuda_from_gpu", &["hello_cuda_from_gpu"])?;
        let hello_cuda_from_gpu = dev
            .get_func("hello_cuda_from_gpu", "hello_cuda_from_gpu")
            .unwrap();
        let cfg = LaunchConfig {
            block_dim: (1, 1, 8),
            grid_dim: (1, 1, 2),
            shared_mem_bytes: 0,
        };
        unsafe {
            hello_cuda_from_gpu.launch(cfg, (0,))?;
        }
    }
    {
        // CPU
        println!("CPU: Hello, CUDA!");
    }
    Ok(())
}

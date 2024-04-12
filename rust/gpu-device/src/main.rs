use nvml_wrapper::Nvml;
fn main() {
    let nvml = Nvml::init().unwrap();
    // Get the first `Device` (GPU) in the system
    let device = nvml.device_by_index(0).unwrap();
    // Get the total energy consumption in millijoules
    let total_energy_millijoules = device.total_energy_consumption().unwrap();

    // Convert total energy consumption from millijoules to kilowatt-hours
    let total_energy_kwh = total_energy_millijoules / 3_600_000_000;

    print!("\x1B[2J\x1B[1;1H");

    println!("Total Energy Consumption: {} kWh", total_energy_kwh);

    println!("GPU_name: {:?}", device.name().unwrap());
    println!("num_fans: {:?}", device.num_fans().unwrap());
    println!("fan_speed: {:?}", device.fan_speed(0).unwrap());
    let memory_info = device.memory_info().unwrap();
    let free_memory_gib = memory_info.free as f64 / 1073741824.0;
    let total_memory_gib = memory_info.total as f64 / 1073741824.0;
    let used_memory_gib = memory_info.used as f64 / 1073741824.0;
    println!(
        "Free Memory (GiB): {:.2} | Total Memory (GiB): {:.2} | Used Memory (GiB): {:.2}",
        free_memory_gib, total_memory_gib, used_memory_gib
    );
}

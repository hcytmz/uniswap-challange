#![warn(unreachable_pub)]
#![deny(unused_must_use, rust_2018_idioms)]

use alloy_primitives::{hex, Address, FixedBytes, Keccak256};
use console::Term;
use fs4::FileExt;
use ocl::{Buffer, Context, Device, MemFlags, Platform, ProQue, Program, Queue};
use parking_lot::Mutex;
use rand::{thread_rng, Rng};
use separator::Separatable;
use std::error::Error;
use std::fmt::Write as _;
use std::fs::{File, OpenOptions};
use std::io::prelude::*;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use terminal_size::{terminal_size, Height};

mod score;
use score::AddressScore;

// workset size (tweak this!)
const WORK_SIZE: u32 = 0x4000000; // max. 0x15400000 to abs. max 0xffffffff

const WORK_FACTOR: u128 = (WORK_SIZE as u128) / 1_000_000;
const CONTROL_CHARACTER: u8 = 0xff;
const MAX_INCREMENTER: u64 = 0xffffffffffff;

const FACTORY_ADDRESS: [u8; 20] = hex!("48E516B34A1274f49457b9C6182097796D0498Cb");
const CALLING_ADDRESS: [u8; 20] = hex!("0000000000000000000000000000000000000000");
const INIT_CODE_HASH: [u8; 32] =
    hex!("94d114296a5af85c1fd2dc039cdaa32f1ed4b0fe0868f02d888bfc91feb645d9");

static KERNEL_SRC: &str = include_str!("./kernels/keccak256.cl");

pub struct Config {
    pub factory_address: [u8; 20],
    pub calling_address: [u8; 20],
    pub init_code_hash: [u8; 32],
    pub gpu_device: u8,
    pub leading_zeroes_threshold: u8,
    pub total_zeroes_threshold: u8,
}

impl Config {
    pub fn from_args(mut args: impl Iterator<Item = String>) -> Result<Self, &'static str> {
        args.next();

        let gpu_device = args
            .next()
            .unwrap_or_else(|| "255".to_string())
            .parse::<u8>()
            .map_err(|_| "invalid gpu device value")?;
        let leading_zeroes_threshold = args
            .next()
            .unwrap_or_else(|| "3".to_string())
            .parse::<u8>()
            .map_err(|_| "invalid leading zeroes threshold value supplied")?;
        let total_zeroes_threshold = args
            .next()
            .unwrap_or_else(|| "5".to_string())
            .parse::<u8>()
            .map_err(|_| "invalid total zeroes threshold value supplied")?;

        if leading_zeroes_threshold > 20 {
            return Err("invalid value for leading zeroes threshold argument. (valid: 0..=20)");
        }
        if total_zeroes_threshold > 20 && total_zeroes_threshold != 255 {
            return Err("invalid value for total zeroes threshold argument. (valid: 0..=20 | 255)");
        }

        Ok(Self {
            factory_address: FACTORY_ADDRESS,
            calling_address: CALLING_ADDRESS,
            init_code_hash: INIT_CODE_HASH,
            gpu_device,
            leading_zeroes_threshold,
            total_zeroes_threshold,
        })
    }
}

pub fn cpu(config: Config) -> Result<(), Box<dyn Error>> {
    let file = output_file();
    let highest_score = Arc::new(Mutex::new(0));

    loop {
        let mut header = [0; 47];
        header[0] = CONTROL_CHARACTER;
        header[1..21].copy_from_slice(&config.factory_address);
        header[21..41].copy_from_slice(&config.calling_address);
        header[41..].copy_from_slice(&FixedBytes::<6>::random()[..]);

        let mut hash_header = Keccak256::new();
        hash_header.update(header);

        let n_jobs = std::thread::available_parallelism().map_or(1, |x| x.get());
        std::thread::scope(|scope| {
            for idx in 0..n_jobs {
                let highest_score = Arc::clone(&highest_score);
                let f = mk_cpu_task(
                    &config,
                    &file,
                    &header,
                    &hash_header,
                    idx,
                    n_jobs,
                    highest_score,
                );
                std::thread::Builder::new()
                    .name(format!("worker-{idx}"))
                    .spawn_scoped(scope, f)
                    .unwrap();
            }
        })
    }
}

fn mk_cpu_task<'a>(
    config: &'a Config,
    file: &'a File,
    header: &'a [u8; 47],
    hash_header: &'a Keccak256,
    idx: usize,
    incr: usize,
    highest_score: Arc<Mutex<u32>>,
) -> impl FnOnce() + 'a {
    move || {
        cpu_task(
            config,
            file,
            header,
            hash_header,
            idx as u64,
            incr as u64,
            highest_score,
        )
    }
}

fn cpu_task(
    config: &Config,
    mut file: &File,
    header: &[u8; 47],
    hash_header: &Keccak256,
    mut salt: u64,
    incr: u64,
    highest_score: Arc<Mutex<u32>>,
) {
    while salt < MAX_INCREMENTER {
        let salt_bytes = salt.to_le_bytes();
        salt += incr;
        let salt_incremented_segment = &salt_bytes[..6];

        let mut hash = hash_header.clone();
        hash.update(salt_incremented_segment);
        hash.update(config.init_code_hash);

        let res = hash.finalize();
        let address = <&Address>::try_from(&res[12..]).unwrap();
        let address_score = AddressScore::calculate(&(*address.0)).score;

        {
            let mut current_highest = highest_score.lock();
            if address_score <= *current_highest || address_score < 70 {
                continue;
            }
            *current_highest = address_score;
        }

        let header_hex_string = hex::encode(header);
        let body_hex_string = hex::encode(salt_incremented_segment);
        let full_salt = format!("0x{}{}", &header_hex_string[42..], &body_hex_string);

        let output = format!("{full_salt} => {address} => {address_score}");
        println!("{output}");

        file.lock_exclusive().expect("Couldn't lock file.");
        writeln!(file, "{output}").expect("Couldn't write to `efficient_addresses.txt` file.");
        file.unlock().expect("Couldn't unlock file.")
    }
}

pub fn gpu(config: Config) -> ocl::Result<()> {
    println!(
        "Setting up experimental OpenCL miner using device {}...",
        config.gpu_device
    );

    let file = output_file();
    let mut found: u64 = 0;
    let mut found_list: Vec<String> = vec![];
    let term = Term::stdout();
    let platform = Platform::new(ocl::core::default_platform()?);
    let device = Device::by_idx_wrap(platform, config.gpu_device as usize)?;
    let context = Context::builder()
        .platform(platform)
        .devices(device)
        .build()?;
    let program = Program::builder()
        .devices(device)
        .src(mk_kernel_src(&config))
        .build(&context)?;
    let queue = Queue::new(&context, device, None)?;
    let ocl_pq = ProQue::new(context, queue, program, Some(WORK_SIZE));
    let mut rng = thread_rng();
    let start_time: f64 = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs_f64();
    let mut rate: f64 = 0.0;
    let mut cumulative_nonce: u64 = 0;
    let mut previous_time: f64 = 0.0;
    let mut work_duration_millis: u64 = 0;

    loop {
        let salt = FixedBytes::<4>::random();
        let message_buffer = Buffer::builder()
            .queue(ocl_pq.queue().clone())
            .flags(MemFlags::new().read_only())
            .len(4)
            .copy_host_slice(&salt[..])
            .build()?;
        let mut nonce: u32 = rng.gen();
        let mut view_buf = [0; 8];
        let mut nonce_buffer = Buffer::builder()
            .queue(ocl_pq.queue().clone())
            .flags(MemFlags::new().read_only())
            .len(1)
            .copy_host_slice(&[nonce])
            .build()?;
        let mut solutions = vec![0u64; 1];
        let solutions_buffer = Buffer::builder()
            .queue(ocl_pq.queue().clone())
            .flags(MemFlags::new().write_only())
            .len(1)
            .copy_host_slice(&solutions)
            .build()?;

        loop {
            let kern = ocl_pq
                .kernel_builder("hashMessage")
                .arg_named("message", None::<&Buffer<u8>>)
                .arg_named("nonce", None::<&Buffer<u32>>)
                .arg_named("solutions", None::<&Buffer<u64>>)
                .build()?;
            kern.set_arg("message", Some(&message_buffer))?;
            kern.set_arg("nonce", Some(&nonce_buffer))?;
            kern.set_arg("solutions", &solutions_buffer)?;
            unsafe { kern.enq()? };

            let mut now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
            let current_time = now.as_secs() as f64;
            let print_output = current_time - previous_time > 0.99;
            previous_time = current_time;

            if print_output {
                term.clear_screen()?;
                let total_runtime = current_time - start_time;
                let total_runtime_hrs = total_runtime as u64 / 3600;
                let total_runtime_mins = (total_runtime as u64 - total_runtime_hrs * 3600) / 60;
                let total_runtime_secs = total_runtime
                    - (total_runtime_hrs * 3600) as f64
                    - (total_runtime_mins * 60) as f64;
                let work_rate: u128 = WORK_FACTOR * cumulative_nonce as u128;
                if total_runtime > 0.0 {
                    rate = 1.0 / total_runtime;
                }
                view_buf.copy_from_slice(&((nonce as u64) << 32).to_le_bytes());
                let height = terminal_size().map(|(_w, Height(h))| h).unwrap_or(10);

                term.write_line(&format!(
                    "total runtime: {}:{:02}:{:02} ({} cycles)\t\t\t\
                     work size per cycle: {}",
                    total_runtime_hrs,
                    total_runtime_mins,
                    total_runtime_secs,
                    cumulative_nonce,
                    WORK_SIZE.separated_string(),
                ))?;
                term.write_line(&format!(
                    "rate: {:.2} million attempts per second\t\t\t\
                     total found this run: {}",
                    work_rate as f64 * rate,
                    found
                ))?;
                term.write_line(&format!(
                    "current search space: {}xxxxxxxx{:08x}\t\t\
                     threshold: {} leading or {} total zeroes",
                    hex::encode(salt),
                    u64::from_be_bytes(view_buf),
                    config.leading_zeroes_threshold,
                    config.total_zeroes_threshold
                ))?;
                let rows = if height < 5 { 1 } else { height as usize - 4 };
                let last_rows: Vec<String> = found_list.iter().cloned().rev().take(rows).collect();
                let ordered: Vec<String> = last_rows.iter().cloned().rev().collect();
                let recently_found = &ordered.join("\n");
                term.write_line(recently_found)?;
            }

            cumulative_nonce += 1;
            let work_start_time_millis = now.as_secs() * 1000 + now.subsec_nanos() as u64 / 1000000;
            if work_duration_millis != 0 {
                std::thread::sleep(std::time::Duration::from_millis(
                    work_duration_millis * 980 / 1000,
                ));
            }
            solutions_buffer.read(&mut solutions).enq()?;
            now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
            work_duration_millis = (now.as_secs() * 1000 + now.subsec_nanos() as u64 / 1000000)
                - work_start_time_millis;

            if solutions[0] != 0 {
                break;
            }
            nonce += 1;
            nonce_buffer = Buffer::builder()
                .queue(ocl_pq.queue().clone())
                .flags(MemFlags::new().read_write())
                .len(1)
                .copy_host_slice(&[nonce])
                .build()?;
        }

        for &solution in &solutions {
            if solution == 0 {
                continue;
            }

            let solution = solution.to_le_bytes();
            let mut solution_message = [0; 85];
            solution_message[0] = CONTROL_CHARACTER;
            solution_message[1..21].copy_from_slice(&config.factory_address);
            solution_message[21..41].copy_from_slice(&config.calling_address);
            solution_message[41..45].copy_from_slice(&salt[..]);
            solution_message[45..53].copy_from_slice(&solution);
            solution_message[53..].copy_from_slice(&config.init_code_hash);

            let res = alloy_primitives::keccak256(solution_message);
            let address = <&Address>::try_from(&res[12..]).unwrap();
            let address_score = AddressScore::calculate(&(*address.0)).score;

            if address_score < 130 {
                continue;
            }

            let output = format!(
                "0x{}{}{} => {} => {}",
                hex::encode(config.calling_address),
                hex::encode(salt),
                hex::encode(solution),
                address,
                address_score
            );

            let show = output.to_string();
            found_list.push(show.to_string());

            file.lock_exclusive().expect("Couldn't lock file.");
            writeln!(&file, "{output}").expect("Couldn't write to `efficient_addresses.txt` file.");
            file.unlock().expect("Couldn't unlock file.");
            found += 1;
        }
    }
}

#[track_caller]
fn output_file() -> File {
    OpenOptions::new()
        .append(true)
        .create(true)
        .read(true)
        .open("efficient_addresses.txt")
        .expect("Could not create or open `efficient_addresses.txt` file.")
}

fn mk_kernel_src(config: &Config) -> String {
    let mut src = String::with_capacity(2048 + KERNEL_SRC.len());

    let factory = config.factory_address.iter();
    let caller = config.calling_address.iter();
    let hash = config.init_code_hash.iter();
    let hash = hash.enumerate().map(|(i, x)| (i + 52, x));
    for (i, x) in factory.chain(caller).enumerate().chain(hash) {
        writeln!(src, "#define S_{} {}u", i + 1, x).unwrap();
    }
    let lz = config.leading_zeroes_threshold;
    writeln!(src, "#define LEADING_ZEROES {lz}").unwrap();
    let tz = config.total_zeroes_threshold;
    writeln!(src, "#define TOTAL_ZEROES {tz}").unwrap();

    src.push_str(KERNEL_SRC);

    src
}

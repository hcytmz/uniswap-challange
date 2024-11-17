# Uniswap Challange

> A Rust tool optimized for the Uniswap challenge to find salts that generate gas-efficient Ethereum addresses.

This project is a modified version of the original `create2crunch` tool by [`0age`](https://github.com/0age/create2crunch). The code has been customized specifically for the Uniswap challenge.

## Installation & Usage

### Docker Setup
1. **Clone the repository**:
   ```sh
   git clone https://github.com/codeesura/uniswap-challange
   cd uniswap-challange
   ```

2. **Build the Docker image**:
   ```sh
   docker build -t uniswap-challange .
   ```

3. **Run the Docker container**:
   ```sh
   docker run -d --name uniswap-container uniswap-challange
   ```
4. **Check Logs**:
   ```sh
   docker logs --follow uniswap-container
   ```

### Local Setup
Alternatively, you can run it locally with Rust installed:
```sh
cargo run --release
```

### Notes
- For ARM-based systems (e.g., Apple Silicon), use:
  ```sh
  cargo run --release --no-default-features
  ```
- This tool can be resource-intensive, especially with GPU usage. Monitor your system’s performance to avoid overheating.
  
## Credits
Modified from the original [`0age/create2crunch`](https://github.com/0age/create2crunch).

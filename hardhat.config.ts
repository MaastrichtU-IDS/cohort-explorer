import { HardhatUserConfig } from "hardhat/config";
import "@nomicfoundation/hardhat-toolbox";
import * as dotenv from "dotenv";

dotenv.config();

const config: HardhatUserConfig = {
  solidity: {
    compilers: [
      {
        version: "0.8.30",
        settings: {
          optimizer: { enabled: true, runs: 50 },
          viaIR: true
        }
      },
      {
        version: "0.8.27",
        settings: {
          optimizer: { enabled: true, runs: 200 },
          viaIR: true
        }
      }
    ],
    overrides: {
      "contracts/HonkVerifier.sol": {
        version: "0.8.27",
        settings: {
          optimizer: { enabled: true, runs: 200 },
          viaIR: true
        }
      }
    }
  },
  networks: {
    hardhat: {
      chainId: 31337,
      mining: {
        auto: true,
        interval: 5000
      },
      allowUnlimitedContractSize: true
    },
    localhost: {
      url: process.env.RPC_URL || "http://hardhat:8545",
      chainId: 31337
    }
  },
  paths: {
    sources: "./contracts",
    tests: "./test",
    cache: "./cache",
    artifacts: "./artifacts"
  }
};

export default config;

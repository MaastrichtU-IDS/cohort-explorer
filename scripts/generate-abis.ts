import * as fs from "fs";
import * as path from "path";

async function main() {
  const contracts = [
    "TrustedForwarder",
    "DUOOntology",
    "InstitutionRegistry",
    "AttestationRegistry",
    "DUOConsentVaultV2",
    "DUOConsentToken",
    "AccessCredentialNFT",
    "DUOAttestationResolver",
    "NullifierRegistry",
    "IdentityRegistry",
    "ZKConsentVerifier",
    "CommitmentTracker",
    "ReputationRegistry",
    "IBISVerifier",
    "HonkVerifier",
    "UserIdentityRegistry",
    "RoleGroupRegistry",
    "RoleAccount",
    "RoleAccountFactory",
    "GasSponsor",
  ];

  const outputDir = path.join(__dirname, "..", "api", "contracts", "abis");

  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
  }

  console.log("Generating ABIs for contracts...\n");

  let generated = 0;
  let missing = 0;

  for (const contractName of contracts) {
    const artifactPath = path.join(
      __dirname,
      "..",
      "artifacts",
      "contracts",
      `${contractName}.sol`,
      `${contractName}.json`
    );

    if (fs.existsSync(artifactPath)) {
      const artifact = JSON.parse(fs.readFileSync(artifactPath, "utf8"));
      const output = {
        contractName: contractName,
        abi: artifact.abi,
        bytecode: artifact.bytecode
      };

      fs.writeFileSync(
        path.join(outputDir, `${contractName}.json`),
        JSON.stringify(output, null, 2)
      );
      console.log(`  ✓ Generated ABI for ${contractName}`);
      generated++;
    } else {
      console.log(`  ✗ Artifact not found for ${contractName}`);
      missing++;
    }
  }

  console.log(`\nDone! ${generated} ABIs generated, ${missing} missing.`);
  console.log("ABIs saved to:", outputDir);
}

main().catch(console.error);

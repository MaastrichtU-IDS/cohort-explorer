import { ethers } from "hardhat";
import * as fs from "fs";
import * as path from "path";
import { seedIcd10Hierarchy } from "./seed-icd10";

async function main() {
  const [deployer] = await ethers.getSigners();
  console.log("Deploying contracts with account:", deployer.address);
  console.log("Account balance:", ethers.formatEther(await ethers.provider.getBalance(deployer.address)));

  const derivationSalt = ethers.keccak256(
    ethers.toUtf8Bytes(process.env.DERIVATION_SALT_SEED || "icare4cvd-blockchain-salt")
  );

  console.log("\n1. Deploying TrustedForwarder...");
  const TrustedForwarder = await ethers.getContractFactory("TrustedForwarder");
  const forwarder = await TrustedForwarder.deploy("IcareTrustedForwarder");
  await forwarder.waitForDeployment();
  const forwarderAddress = await forwarder.getAddress();
  console.log("   TrustedForwarder:", forwarderAddress);

  console.log("\n2. Deploying DUOOntology...");
  const DUOOntology = await ethers.getContractFactory("DUOOntology");
  const ontology = await DUOOntology.deploy();
  await ontology.waitForDeployment();
  const ontologyAddress = await ontology.getAddress();
  console.log("   DUOOntology:", ontologyAddress);
  await seedIcd10Hierarchy(ontology);

  console.log("\n3. Deploying InstitutionRegistry...");
  const InstitutionRegistry = await ethers.getContractFactory("InstitutionRegistry");
  const institutionRegistry = await InstitutionRegistry.deploy();
  await institutionRegistry.waitForDeployment();
  const institutionRegistryAddress = await institutionRegistry.getAddress();
  console.log("   InstitutionRegistry:", institutionRegistryAddress);

  console.log("\n4. Deploying AttestationRegistry...");
  const AttestationRegistry = await ethers.getContractFactory("AttestationRegistry");
  const attestationRegistry = await AttestationRegistry.deploy();
  await attestationRegistry.waitForDeployment();
  const attestationRegistryAddress = await attestationRegistry.getAddress();
  console.log("   AttestationRegistry:", attestationRegistryAddress);

  console.log("\n5. Deploying DUOConsentVaultV2...");
  const DUOConsentVaultV2 = await ethers.getContractFactory("DUOConsentVaultV2");
  const consentVaultV2 = await DUOConsentVaultV2.deploy(
    ontologyAddress,
    attestationRegistryAddress,
    institutionRegistryAddress
  );
  await consentVaultV2.waitForDeployment();
  const consentVaultV2Address = await consentVaultV2.getAddress();
  console.log("   DUOConsentVaultV2:", consentVaultV2Address);

  console.log("\n6. Deploying DUOConsentToken...");
  const DUOConsentToken = await ethers.getContractFactory("DUOConsentToken");
  const consentToken = await DUOConsentToken.deploy(
    ontologyAddress,
    attestationRegistryAddress
  );
  await consentToken.waitForDeployment();
  const consentTokenAddress = await consentToken.getAddress();
  console.log("   DUOConsentToken:", consentTokenAddress);

  console.log("\n7. Deploying AccessCredentialNFT...");
  const AccessCredentialNFT = await ethers.getContractFactory("AccessCredentialNFT");
  const accessCredential = await AccessCredentialNFT.deploy(consentTokenAddress);
  await accessCredential.waitForDeployment();
  const accessCredentialAddress = await accessCredential.getAddress();
  console.log("   AccessCredentialNFT:", accessCredentialAddress);

  console.log("\n8. Deploying DUOAttestationResolver...");
  const easAddress = process.env.EAS_ADDRESS || ethers.ZeroAddress;
  const DUOAttestationResolver = await ethers.getContractFactory("DUOAttestationResolver");
  const attestationResolver = await DUOAttestationResolver.deploy(easAddress);
  await attestationResolver.waitForDeployment();
  const attestationResolverAddress = await attestationResolver.getAddress();
  console.log("   DUOAttestationResolver:", attestationResolverAddress);

  console.log("\n9. Deploying NullifierRegistry...");
  const NullifierRegistry = await ethers.getContractFactory("NullifierRegistry");
  const nullifierRegistry = await NullifierRegistry.deploy(deployer.address);
  await nullifierRegistry.waitForDeployment();
  const nullifierRegistryAddress = await nullifierRegistry.getAddress();
  console.log("   NullifierRegistry:", nullifierRegistryAddress);

  console.log("\n10. Deploying IdentityRegistry...");
  const IdentityRegistry = await ethers.getContractFactory("IdentityRegistry");
  const identityRegistry = await IdentityRegistry.deploy(deployer.address);
  await identityRegistry.waitForDeployment();
  const identityRegistryAddress = await identityRegistry.getAddress();
  console.log("   IdentityRegistry:", identityRegistryAddress);

  console.log("\n11. Deploying ZKConsentVerifier...");
  const ZKConsentVerifier = await ethers.getContractFactory("ZKConsentVerifier");
  const zkVerifier = await ZKConsentVerifier.deploy(
    deployer.address,
    nullifierRegistryAddress,
    identityRegistryAddress,
    deployer.address,
    deployer.address
  );
  await zkVerifier.waitForDeployment();
  const zkVerifierAddress = await zkVerifier.getAddress();
  console.log("   ZKConsentVerifier:", zkVerifierAddress);

  const VERIFIER_ROLE = ethers.keccak256(ethers.toUtf8Bytes("VERIFIER_ROLE"));
  await nullifierRegistry.grantRole(VERIFIER_ROLE, zkVerifierAddress);
  const OPERATOR_ROLE = ethers.keccak256(ethers.toUtf8Bytes("OPERATOR_ROLE"));
  await identityRegistry.grantRole(OPERATOR_ROLE, zkVerifierAddress);

  console.log("\n12. Deploying CommitmentTracker...");
  const CommitmentTracker = await ethers.getContractFactory("CommitmentTracker");
  const commitmentTracker = await CommitmentTracker.deploy();
  await commitmentTracker.waitForDeployment();
  const commitmentTrackerAddress = await commitmentTracker.getAddress();
  console.log("   CommitmentTracker:", commitmentTrackerAddress);

  console.log("\n13. Deploying ReputationRegistry...");
  const ReputationRegistry = await ethers.getContractFactory("ReputationRegistry");
  const reputationRegistry = await ReputationRegistry.deploy();
  await reputationRegistry.waitForDeployment();
  const reputationRegistryAddress = await reputationRegistry.getAddress();
  console.log("   ReputationRegistry:", reputationRegistryAddress);

  const REPUTATION_UPDATER_ROLE = ethers.keccak256(ethers.toUtf8Bytes("REPUTATION_UPDATER"));
  await reputationRegistry.grantRole(REPUTATION_UPDATER_ROLE, commitmentTrackerAddress);

  const ATT_IRB = ethers.keccak256(ethers.toUtf8Bytes("IRB_APPROVAL"));
  const ATT_GEO = ethers.keccak256(ethers.toUtf8Bytes("GEOGRAPHIC"));
  await attestationRegistry.addTrustedAttestor(ATT_IRB, deployer.address);
  await attestationRegistry.addTrustedAttestor(ATT_GEO, deployer.address);

  console.log("\n14. Deploying IBISVerifier...");
  const IBISVerifier = await ethers.getContractFactory("IBISVerifier");
  const ibisVerifier = await IBISVerifier.deploy();
  await ibisVerifier.waitForDeployment();
  const ibisVerifierAddress = await ibisVerifier.getAddress();
  console.log("   IBISVerifier:", ibisVerifierAddress);

  console.log("\n14a. Deploying HonkVerifier (chain_position circuit)...");
  const ZKTranscriptLib = await ethers.getContractFactory("contracts/HonkVerifier.sol:ZKTranscriptLib");
  const zkTranscriptLib = await ZKTranscriptLib.deploy();
  await zkTranscriptLib.waitForDeployment();
  const zkTranscriptLibAddress = await zkTranscriptLib.getAddress();
  console.log("   ZKTranscriptLib:", zkTranscriptLibAddress);
  const HonkVerifier = await ethers.getContractFactory("HonkVerifier", {
    libraries: { "contracts/HonkVerifier.sol:ZKTranscriptLib": zkTranscriptLibAddress },
  });
  const honkVerifier = await HonkVerifier.deploy();
  await honkVerifier.waitForDeployment();
  const honkVerifierAddress = await honkVerifier.getAddress();
  console.log("   HonkVerifier:", honkVerifierAddress);
  await ibisVerifier.setChainPositionVerifier(honkVerifierAddress);
  console.log("   Wired HonkVerifier into IBISVerifier");

  await consentVaultV2.setIBISVerifier(ibisVerifierAddress);
  console.log("   Wired IBISVerifier into DUOConsentVaultV2");

  console.log("\n14b. Deploying UserIdentityRegistry...");
  const UserIdentityRegistry = await ethers.getContractFactory("UserIdentityRegistry");
  const userIdentityRegistry = await UserIdentityRegistry.deploy(deployer.address);
  await userIdentityRegistry.waitForDeployment();
  const userIdentityRegistryAddress = await userIdentityRegistry.getAddress();
  console.log("   UserIdentityRegistry:", userIdentityRegistryAddress);

  await userIdentityRegistry.setIBISVerifier(ibisVerifierAddress);
  console.log("   Wired IBISVerifier into UserIdentityRegistry");

  console.log("\n15. Deploying RoleGroupRegistry...");
  const RoleGroupRegistry = await ethers.getContractFactory("RoleGroupRegistry");
  const roleGroupRegistry = await RoleGroupRegistry.deploy(deployer.address);
  await roleGroupRegistry.waitForDeployment();
  const roleGroupRegistryAddress = await roleGroupRegistry.getAddress();
  console.log("   RoleGroupRegistry:", roleGroupRegistryAddress);

  console.log("\n16. Deploying RoleAccountFactory...");
  const RoleAccountFactory = await ethers.getContractFactory("RoleAccountFactory");
  const roleAccountFactory = await RoleAccountFactory.deploy();
  await roleAccountFactory.waitForDeployment();
  const roleAccountFactoryAddress = await roleAccountFactory.getAddress();
  console.log("   RoleAccountFactory:", roleAccountFactoryAddress);

  await consentVaultV2.setRoleInfra(roleGroupRegistryAddress, roleAccountFactoryAddress);
  console.log("   Wired role infra into DUOConsentVaultV2");

  await attestationRegistry.setRoleInfra(roleGroupRegistryAddress, roleAccountFactoryAddress);
  console.log("   Wired role infra into AttestationRegistry (enables recordCommitmentWithSignature)");

  const ROLE_OPERATOR = ethers.keccak256(ethers.toUtf8Bytes("ROLE_OPERATOR"));
  await roleGroupRegistry.grantRole(ROLE_OPERATOR, deployer.address);
  console.log("   Granted ROLE_OPERATOR to relayer/deployer");

  await roleGroupRegistry.setUserIdentityRegistry(userIdentityRegistryAddress);
  console.log("   Wired UserIdentityRegistry into RoleGroupRegistry");

  console.log("\n15. Deploying GasSponsor...");
  const GasSponsor = await ethers.getContractFactory("GasSponsor");
  const gasSponsor = await GasSponsor.deploy(deployer.address);
  await gasSponsor.waitForDeployment();
  const gasSponsorAddress = await gasSponsor.getAddress();
  console.log("   GasSponsor:", gasSponsorAddress);

  const RELAY_ROLE = ethers.keccak256(ethers.toUtf8Bytes("RELAY_ROLE"));
  await gasSponsor.grantRole(RELAY_ROLE, deployer.address);

  const deployments = {
    chainId: (await ethers.provider.getNetwork()).chainId.toString(),
    contracts: {
      trustedForwarder: forwarderAddress,
      duoOntology: ontologyAddress,
      institutionRegistry: institutionRegistryAddress,
      attestationRegistry: attestationRegistryAddress,
      duoConsentVaultV2: consentVaultV2Address,
      duoConsentToken: consentTokenAddress,
      accessCredentialNFT: accessCredentialAddress,
      duoAttestationResolver: attestationResolverAddress,
      nullifierRegistry: nullifierRegistryAddress,
      identityRegistry: identityRegistryAddress,
      zkConsentVerifier: zkVerifierAddress,
      commitmentTracker: commitmentTrackerAddress,
      reputationRegistry: reputationRegistryAddress,
      ibisVerifier: ibisVerifierAddress,
      honkVerifier: honkVerifierAddress,
      userIdentityRegistry: userIdentityRegistryAddress,
      roleGroupRegistry: roleGroupRegistryAddress,
      roleAccountFactory: roleAccountFactoryAddress,
      gasSponsor: gasSponsorAddress,
    },
    derivationSalt,
    deployer: deployer.address,
    timestamp: new Date().toISOString()
  };

  const deploymentsDir = path.join(__dirname, "..", "deployments");
  if (!fs.existsSync(deploymentsDir)) fs.mkdirSync(deploymentsDir, { recursive: true });
  fs.writeFileSync(path.join(deploymentsDir, "deployments.json"), JSON.stringify(deployments, null, 2));
  fs.writeFileSync(path.join(__dirname, "..", "deployments.json"), JSON.stringify(deployments, null, 2));

  const envContent = `# Contract Addresses (auto-generated)
TRUSTED_FORWARDER_ADDRESS=${forwarderAddress}
DUO_ONTOLOGY_ADDRESS=${ontologyAddress}
INSTITUTION_REGISTRY_ADDRESS=${institutionRegistryAddress}
ATTESTATION_REGISTRY_ADDRESS=${attestationRegistryAddress}
DUO_CONSENT_VAULT_V2_ADDRESS=${consentVaultV2Address}
DUO_CONSENT_TOKEN_ADDRESS=${consentTokenAddress}
ACCESS_CREDENTIAL_NFT_ADDRESS=${accessCredentialAddress}
DUO_ATTESTATION_RESOLVER_ADDRESS=${attestationResolverAddress}
NULLIFIER_REGISTRY_ADDRESS=${nullifierRegistryAddress}
IDENTITY_REGISTRY_ADDRESS=${identityRegistryAddress}
ZK_CONSENT_VERIFIER_ADDRESS=${zkVerifierAddress}
COMMITMENT_TRACKER_ADDRESS=${commitmentTrackerAddress}
REPUTATION_REGISTRY_ADDRESS=${reputationRegistryAddress}
IBIS_VERIFIER_ADDRESS=${ibisVerifierAddress}
HONK_VERIFIER_ADDRESS=${honkVerifierAddress}
USER_IDENTITY_REGISTRY_ADDRESS=${userIdentityRegistryAddress}
ROLE_GROUP_REGISTRY_ADDRESS=${roleGroupRegistryAddress}
ROLE_ACCOUNT_FACTORY_ADDRESS=${roleAccountFactoryAddress}
GAS_SPONSOR_ADDRESS=${gasSponsorAddress}
DERIVATION_SALT=${derivationSalt}
`;
  fs.writeFileSync(path.join(deploymentsDir, "contracts.env"), envContent);
  console.log("\nDeployments saved to deployments/deployments.json");
  console.log("Environment file written to deployments/contracts.env");
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});

import { ethers } from "hardhat";
import * as fs from "fs";
import * as path from "path";

const HIERARCHY_CANDIDATES = [
  path.join(__dirname, "..", "api", "services", "ontology", "icd10_hierarchy.json"),
  path.join(__dirname, "icd10_hierarchy.json"),
];

const BATCH_SIZE = 50;

function readHierarchy(): any {
  const p = HIERARCHY_CANDIDATES.find((c) => fs.existsSync(c));
  if (!p) throw new Error("icd10_hierarchy.json not found in " + HIERARCHY_CANDIDATES.join(", "));
  return JSON.parse(fs.readFileSync(p, "utf8"));
}

export async function seedIcd10Hierarchy(ontology: any) {
  const raw = readHierarchy();
  const parents: Record<string, string> = raw.parents;

  const children: string[] = [];
  const parentHashes: string[] = [];
  for (const [child, parent] of Object.entries(parents)) {
    children.push(ethers.keccak256(ethers.toUtf8Bytes(child)));
    parentHashes.push(ethers.keccak256(ethers.toUtf8Bytes(parent)));
  }

  console.log(`   Seeding ${children.length} ICD-10 hierarchy edges...`);
  for (let i = 0; i < children.length; i += BATCH_SIZE) {
    const c = children.slice(i, i + BATCH_SIZE);
    const p = parentHashes.slice(i, i + BATCH_SIZE);
    const tx = await ontology.setDiseaseHierarchyBatch(c, p);
    await tx.wait();
    console.log(`   Seeded edges ${i + 1}-${i + c.length}`);
  }
  console.log("   ICD-10 hierarchy seeded");
}

async function main() {
  const paths = [
    path.join(__dirname, "..", "deployments", "deployments.json"),
    path.join(__dirname, "..", "deployments.json"),
  ];
  const dpath = paths.find((p) => fs.existsSync(p));
  if (!dpath) throw new Error("deployments.json not found; deploy first");

  const deployments = JSON.parse(fs.readFileSync(dpath, "utf8"));
  const address = deployments.contracts?.duoOntology;
  if (!address) throw new Error("duoOntology address missing from deployments.json");

  console.log("Seeding ICD-10 hierarchy into DUOOntology:", address);
  const ontology = await ethers.getContractAt("DUOOntology", address);
  await seedIcd10Hierarchy(ontology);
}

if (require.main === module) {
  main().catch((error) => {
    console.error(error);
    process.exitCode = 1;
  });
}

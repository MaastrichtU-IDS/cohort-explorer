import {execFileSync} from 'child_process';
import {createHash} from 'crypto';
import {copyFileSync, mkdirSync, readFileSync, writeFileSync} from 'fs';
import path from 'path';

import {expect, test, type APIResponse, type Download, type Page, type Request} from '@playwright/test';

const browserUrl = process.env.DEMO_BROWSER_URL || 'http://localhost:3001';
const apiUrl = process.env.DEMO_API_URL || 'http://localhost:3000';
const packDir = process.env.DEMO_BROWSER_PACK;
const evidenceDir = process.env.DEMO_BROWSER_EVIDENCE || path.resolve(__dirname, '../../artifacts/browser-demo');
const adminEmail = 'nikolas.molyndris@decentriq.ch';
const analystEmail = 'browser.analyst@example.test';
const roomName = 'Cohort Explorer Local Browser Acceptance';
const researchQuestion = 'Can harmonized synthetic heart-failure cohorts support reproducible aggregate analysis?';
const expectedResultHash = '0b4a24cf910202d0dbaf7fc8ebc445d0bc6c2b731045df1d52f266497160cd32';

if (!packDir) {
  throw new Error('DEMO_BROWSER_PACK must point to the immutable pack printed by make demo-browser-ready');
}

const manifest = JSON.parse(readFileSync(path.join(packDir, 'manifest.json'), 'utf8'));

const dictionaryPath = (cohortId: string): string =>
  path.join(packDir, 'cohorts', cohortId, `${cohortId}_datadictionary.csv`);

const sha256 = (content: Buffer): string => createHash('sha256').update(content).digest('hex');

async function downloadBuffer(download: Download): Promise<Buffer> {
  const downloadedPath = await download.path();
  if (!downloadedPath) throw new Error(`Playwright did not retain ${download.suggestedFilename()}`);
  return readFileSync(downloadedPath);
}

async function responseJson(response: APIResponse, expectedStatus = 200): Promise<any> {
  expect(response.status(), `API response from ${response.url()}`).toBe(expectedStatus);
  return response.json();
}

function zipMembers(archivePath: string): string[] {
  const output = execFileSync(
    'python3',
    ['-c', 'import json,sys,zipfile; print(json.dumps(zipfile.ZipFile(sys.argv[1]).namelist()))', archivePath],
    {encoding: 'utf8'}
  );
  return JSON.parse(output);
}

function zipJson(archivePath: string, member: string): any {
  const output = execFileSync(
    'python3',
    [
      '-c',
      'import sys,zipfile; sys.stdout.buffer.write(zipfile.ZipFile(sys.argv[1]).read(sys.argv[2]))',
      archivePath,
      member
    ],
    {encoding: 'utf8'}
  );
  return JSON.parse(output);
}

function zipMemberSha256(archivePath: string, member: string): string {
  return execFileSync(
    'python3',
    [
      '-c',
      'import hashlib,sys,zipfile; print(hashlib.sha256(zipfile.ZipFile(sys.argv[1]).read(sys.argv[2])).hexdigest())',
      archivePath,
      member
    ],
    {encoding: 'utf8'}
  ).trim();
}

async function checkpoint(page: Page, filename: string): Promise<void> {
  mkdirSync(evidenceDir, {recursive: true});
  await page.screenshot({path: path.join(evidenceDir, filename), fullPage: true, animations: 'disabled'});
}

async function currentMetadata(page: Page): Promise<any> {
  return responseJson(await page.request.get(`${apiUrl}/cohorts-metadata`));
}

async function uploadDictionary(page: Page, cohortId: string, expectEmptyBeforeUpload: boolean): Promise<void> {
  await page.goto(`${browserUrl}/upload`);
  await page.locator('#upload-cohort-select').selectOption(cohortId);
  await page.locator('#metadata-file').setInputFiles(dictionaryPath(cohortId));

  const validationResponsePromise = page.waitForResponse(
    response => response.url() === `${apiUrl}/validate-cohort-dictionary` && response.request().method() === 'POST'
  );
  await page.locator('#validate-dictionary').click();
  expect((await validationResponsePromise).status()).toBe(200);
  await expect(page.locator('#validation-results')).toContainText('Validation successful');

  const metadataAfterValidation = await currentMetadata(page);
  if (expectEmptyBeforeUpload) expect(metadataAfterValidation[cohortId].variables).toEqual({});

  await page.locator('#validation-results').getByRole('button', {name: 'Got it'}).click();
  const uploadResponsePromise = page.waitForResponse(
    response => response.url() === `${apiUrl}/upload-cohort` && response.request().method() === 'POST'
  );
  await page.locator('#upload-dictionary').click();
  expect((await uploadResponsePromise).status()).toBe(200);
  await expect(page.getByTestId('upload-dcr-step-title')).toHaveText('Step 2: Create Local Synthetic AADCR Simulation');
  await expect(page.getByTestId('upload-dcr-provider-copy')).toHaveAttribute('data-provider', 'aadcrv2');
  await expect(page.getByTestId('upload-dcr-provider-warning')).toContainText(
    'does not provide a confidential-computing or production security boundary'
  );
  await expect(
    page
      .locator('[role="alert"]')
      .filter({hasText: /upload/i})
      .first()
  ).toBeVisible();
}

async function openCohort(page: Page, cohortId: string): Promise<void> {
  const card = page.getByTestId(`cohort-${cohortId}`);
  await card.scrollIntoViewIfNeeded();
  if (!(await card.getByRole('button', {name: 'Variables List'}).isVisible())) {
    await card.locator('.collapse-title').click();
  }
  await expect(card.getByRole('button', {name: 'Variables List'})).toBeVisible();
}

async function addCohortToDcr(page: Page, cohortId: string, expectedCount: number): Promise<void> {
  await openCohort(page, cohortId);
  const card = page.getByTestId(`cohort-${cohortId}`);
  await card.getByRole('button', {name: 'Add to DCR'}).click();
  await expect(page.getByTestId('dcr-launcher')).toContainText(String(expectedCount), {timeout: 5_000});
}

test('complete local AADCR journey preserves metadata and returns one aggregate result', async ({page}, testInfo) => {
  const expectedPreAuthConsoleError = 'Error fetching data in cache worker: Not authenticated';
  let monitorAuthenticatedJourney = false;
  const preAuthConsoleErrors: string[] = [];
  const consoleErrors: string[] = [];
  const pageErrors: string[] = [];
  const externalRequests: string[] = [];
  const externalWebSockets: string[] = [];
  const preAuthFailedLocalResponses: string[] = [];
  const approvedLocalFailures: string[] = [];
  const failedLocalResponses: string[] = [];
  const preAuthMetadataRequests = new Set<Request>();
  let loginStarted = false;
  let allowInvalidDictionaryValidation = false;

  page.on('console', message => {
    if (message.type() !== 'error') return;
    if (monitorAuthenticatedJourney) {
      consoleErrors.push(message.text());
      return;
    }
    preAuthConsoleErrors.push(message.text());
  });
  page.on('pageerror', error => pageErrors.push(error.message));
  page.context().on('request', request => {
    const url = new URL(request.url());
    if (
      !loginStarted &&
      url.pathname === '/cohorts-metadata' &&
      request.method() === 'GET' &&
      (url.hostname === 'localhost' || url.hostname === '127.0.0.1')
    ) {
      preAuthMetadataRequests.add(request);
    }
    if (
      (url.protocol === 'http:' || url.protocol === 'https:') &&
      url.hostname !== 'localhost' &&
      url.hostname !== '127.0.0.1'
    ) {
      externalRequests.push(request.url());
    }
  });
  page.on('websocket', socket => {
    const url = new URL(socket.url());
    if (url.hostname !== 'localhost' && url.hostname !== '127.0.0.1') {
      externalWebSockets.push(socket.url());
    }
  });
  page.on('response', response => {
    const url = new URL(response.url());
    if ((url.hostname === 'localhost' || url.hostname === '127.0.0.1') && response.status() >= 400) {
      if (monitorAuthenticatedJourney) {
        if (
          allowInvalidDictionaryValidation &&
          response.status() === 422 &&
          response.request().method() === 'POST' &&
          url.pathname === '/validate-cohort-dictionary'
        ) {
          approvedLocalFailures.push('422 POST /validate-cohort-dictionary');
        } else {
          failedLocalResponses.push(`${response.status()} ${response.request().method()} ${response.url()}`);
        }
      } else {
        preAuthFailedLocalResponses.push(`${response.status()} ${response.request().method()} ${url.pathname}`);
      }
    }
  });

  await page.goto(browserUrl);
  await expect(page.getByText('Login', {exact: true})).toBeVisible();
  const authenticatedMetadataResponse = page.waitForResponse(response => {
    const url = new URL(response.url());
    return url.pathname === '/cohorts-metadata' && response.request().method() === 'GET' && response.status() === 200;
  });
  const providerResponse = page.waitForResponse(response => {
    const url = new URL(response.url());
    return url.pathname === '/api/dcr/provider' && response.request().method() === 'GET' && response.status() === 200;
  });
  loginStarted = true;
  await page.getByText('Login', {exact: true}).click();
  await page.waitForURL(`${browserUrl}/**`);
  await expect(page.getByText('Logout', {exact: true})).toBeVisible();
  expect(preAuthMetadataRequests.size).toBeGreaterThan(0);
  expect(preAuthMetadataRequests.size).toBeLessThanOrEqual(2);
  const preAuthMetadataResponses = await Promise.all([...preAuthMetadataRequests].map(request => request.response()));
  for (const response of preAuthMetadataResponses) {
    expect(response).not.toBeNull();
    expect(response?.status()).toBe(401);
  }
  expect((await authenticatedMetadataResponse).status()).toBe(200);
  expect((await providerResponse).status()).toBe(200);
  await expect.poll(() => preAuthConsoleErrors.length).toBeGreaterThan(0);
  expect(preAuthConsoleErrors.length).toBeLessThanOrEqual(2);
  expect(new Set(preAuthConsoleErrors)).toEqual(new Set([expectedPreAuthConsoleError]));
  expect(preAuthFailedLocalResponses.length).toBeGreaterThan(0);
  expect(preAuthFailedLocalResponses.length).toBeLessThanOrEqual(2);
  expect(new Set(preAuthFailedLocalResponses)).toEqual(new Set(['401 GET /cohorts-metadata']));
  monitorAuthenticatedJourney = true;

  const initialMetadata = await currentMetadata(page);
  expect(initialMetadata.userEmail).toBe(adminEmail);
  expect(initialMetadata['TIME-CHF'].variables).toEqual({});
  expect(initialMetadata['GISSI-HF'].variables).toEqual({});
  expect(initialMetadata['TIME-CHF'].study_participants).toBe('2500');
  expect(initialMetadata['GISSI-HF'].study_participants).toBe('2500');

  const admin = await responseJson(await page.request.get(`${apiUrl}/admin/check`));
  expect(admin).toEqual({is_admin: true});
  await page.goto(`${browserUrl}/admin-settings`);
  await expect(page.getByRole('heading', {name: 'Admin Settings'})).toBeVisible();
  await checkpoint(page, '01-login-admin.png');

  const initialRooms = await responseJson(await page.request.get(`${apiUrl}/my-dcrs`));
  expect(initialRooms.email).toBe(adminEmail);
  expect(initialRooms.dcrs).toEqual([]);
  const initialMappings = await responseJson(
    await page.request.post(`${apiUrl}/api/get-available-mapping-files`, {data: ['TIME-CHF', 'GISSI-HF']})
  );
  expect(initialMappings.available_mappings).toEqual([]);
  const initialMappingActivity = await responseJson(await page.request.get(`${apiUrl}/api/mapping-activity-log`));
  expect(initialMappingActivity.total).toBe(0);

  const analysisCheck = await page.request.get(`${browserUrl}/api/check-analysis-folder/TIME-CHF`);
  expect(analysisCheck.status()).toBe(200);
  expect(await analysisCheck.json()).toMatchObject({exists: true, cohortId: 'TIME-CHF'});
  const edaResponse = await page.request.get(`${browserUrl}/api/cohort-eda-output/TIME-CHF`);
  expect(edaResponse.status()).toBe(200);
  expect(await edaResponse.json()).toHaveProperty('age');
  const graphResponse = await page.request.get(`${browserUrl}/api/variable-graph/TIME-CHF/age`);
  expect(graphResponse.status()).toBe(200);
  expect(graphResponse.headers()['content-type']).toContain('image/png');
  expect((await graphResponse.body()).subarray(1, 4).toString()).toBe('PNG');

  await uploadDictionary(page, 'TIME-CHF', true);
  await uploadDictionary(page, 'GISSI-HF', true);
  await checkpoint(page, '02-dictionaries-uploaded.png');

  const uploadedMetadata = await currentMetadata(page);
  for (const cohortId of ['TIME-CHF', 'GISSI-HF']) {
    expect(Object.keys(uploadedMetadata[cohortId].variables)).toHaveLength(35);
    expect(uploadedMetadata[cohortId].study_participants).toBe(String(manifest.cohorts[cohortId].row_count));
    expect(uploadedMetadata[cohortId].can_edit).toBe(true);
  }

  await page.goto(browserUrl);
  const expectedStats: Record<string, string> = {
    'Registered Cohorts': '2',
    'Cohorts with Uploaded Metadata': '2',
    'Cohorts with Aggregate Data Added': '2',
    'Total Patients Across All Cohorts': '5,000',
    'Patients in Cohorts with Uploaded Metadata': '5,000',
    'Variables in Cohorts with Uploaded Metadata': '70'
  };
  for (const [title, value] of Object.entries(expectedStats)) {
    const stat = page.locator('.stat-title').getByText(title, {exact: true}).locator('..');
    await expect(stat.locator('.stat-value')).toHaveText(value, {
      timeout: 15_000
    });
  }

  await page.goto(`${browserUrl}/cohorts`);
  for (const cohortId of ['TIME-CHF', 'GISSI-HF']) {
    const card = page.getByTestId(`cohort-${cohortId}`);
    await expect(card).toBeVisible();
    await expect(card).toContainText('Synthetic iCARE4CVD Demo Consortium');
    await expect(card).toContainText('2500');
  }

  const studyFilter = page.getByTestId('metadata-filter-study_design');
  const prospectiveStudy = studyFilter
    .getByText('Prospective synthetic cohort study (1)', {exact: true})
    .locator('..')
    .getByRole('checkbox');
  await prospectiveStudy.check();
  await expect(page.getByTestId('cohort-TIME-CHF')).toBeVisible();
  await expect(page.getByTestId('cohort-GISSI-HF')).toHaveCount(0);
  await prospectiveStudy.uncheck();

  const providerFilter = page.getByTestId('metadata-filter-institution');
  const timeProvider = providerFilter
    .getByText('Synthetic iCARE4CVD Demo Consortium - TIME-CHF Site (1)', {exact: true})
    .locator('..')
    .getByRole('checkbox');
  await timeProvider.check();
  await expect(page.getByTestId('cohort-TIME-CHF')).toBeVisible();
  await expect(page.getByTestId('cohort-GISSI-HF')).toHaveCount(0);
  await timeProvider.uncheck();

  const cohortSearch = page.getByTestId('cohort-search');
  await page.getByRole('button', {name: /Cohorts Metadata/}).click();
  await page.getByRole('button', {name: /OR Search/}).click();
  await cohortSearch.fill('TIME GISSI');
  await expect(page.getByText(/Search matched 2 cohorts metadata/)).toBeVisible();
  await expect(page.getByTestId('cohort-TIME-CHF')).toBeVisible();
  await expect(page.getByTestId('cohort-GISSI-HF')).toBeVisible();
  await page.getByRole('button', {name: /AND Search/}).click();
  await expect(page.getByText(/Search matched 0 cohorts metadata/)).toBeVisible();
  await expect(page.getByText('0/2 cohorts', {exact: true})).toBeVisible();
  await page.getByRole('button', {name: /Exact Phrase/}).click();
  await cohortSearch.fill('Cohort study');
  await expect(page.getByText(/Search matched 2 cohorts metadata/)).toBeVisible();
  await expect(page.getByText('study design', {exact: true}).first()).toBeVisible();
  await cohortSearch.fill('Synthetic iCARE4CVD Demo Consortium');
  await expect(page.getByText(/Search matched 2 cohorts metadata/)).toBeVisible();
  await expect(page.getByText('institution', {exact: true}).first()).toBeVisible();
  await page.getByRole('button', {name: '✕ Clear'}).click();

  await openCohort(page, 'TIME-CHF');
  const timeCard = page.getByTestId('cohort-TIME-CHF');
  const metadataDownloadPromise = page.waitForEvent('download');
  await timeCard.getByRole('button', {name: 'Download Metadata'}).click();
  const metadataDownload = await metadataDownloadPromise;
  expect(metadataDownload.suggestedFilename()).toMatch(/TIME-CHF.*\.csv/i);
  expect((await downloadBuffer(metadataDownload)).toString('utf8')).toContain('VARIABLENAME,VARIABLELABEL');

  await timeCard.getByRole('button', {name: 'Variables List'}).click();
  await expect(timeCard.getByText('35 variables', {exact: true})).toBeVisible();
  const timeVariables = Object.values(uploadedMetadata['TIME-CHF'].variables) as any[];
  const assertVariableFilter = async (filterId: string, option: string, expectedCount: number) => {
    const checkbox = timeCard
      .getByTestId(`metadata-filter-${filterId}`)
      .getByText(`${option} (${expectedCount})`, {exact: true})
      .locator('..')
      .getByRole('checkbox');
    await checkbox.check();
    await expect(timeCard.getByText(`${expectedCount}/35 variables`, {exact: true})).toBeVisible();
    await checkbox.uncheck();
    await expect(timeCard.getByText('35 variables', {exact: true})).toBeVisible();
  };
  await assertVariableFilter(
    'omop_domain',
    'measurement',
    timeVariables.filter(variable => variable.omop_domain === 'measurement').length
  );
  await assertVariableFilter('var_type', 'STR', timeVariables.filter(variable => variable.var_type === 'STR').length);
  await assertVariableFilter(
    'categorical',
    '4+ categories',
    timeVariables.filter(variable => variable.categories.length >= 4).length
  );
  await assertVariableFilter(
    'visits',
    'follow-up 1 year',
    timeVariables.filter(variable => variable.visits === 'follow-up 1 year').length
  );

  const ehrCount = timeVariables.filter(variable =>
    String(variable.source_name)
      .split('|')
      .map(source => source.trim())
      .includes('EHR')
  ).length;
  const ehrSourceTab = timeCard.getByTestId('source-tab-TIME-CHF-EHR');
  await expect(ehrSourceTab).toContainText('Electronic Health Record');
  await expect(ehrSourceTab).toContainText(String(ehrCount));
  await ehrSourceTab.click();
  await expect(timeCard.locator('[data-testid^="variable-TIME-CHF-"]')).toHaveCount(ehrCount);
  await timeCard.getByTestId('source-tab-TIME-CHF-all').click();
  await expect(timeCard.locator('[data-testid^="variable-TIME-CHF-"]')).toHaveCount(35);

  await timeCard.getByRole('button', {name: 'Show Outcome Variables'}).click();
  await expect(timeCard.getByText('1/35 variables', {exact: true})).toBeVisible();
  await expect(page.getByTestId('variable-TIME-CHF-hf_hosp')).toContainText(
    'emergency hospital admission for heart failure'
  );
  await timeCard.getByRole('button', {name: 'Show All Variables'}).click();
  await expect(timeCard.getByText('35 variables', {exact: true})).toBeVisible();

  await page.getByRole('button', {name: /Variables Information/}).click();
  await page.getByRole('button', {name: /Exact Phrase/}).click();
  await page.getByTestId('cohort-search').fill('age');
  await expect(page.getByRole('button', {name: /Show equivalent variable names/})).toBeVisible();
  await page.getByRole('button', {name: /Show equivalent variable names/}).click();
  const standardCodeGroup = page.getByText('Standard code:').locator('..');
  await expect(standardCodeGroup).toContainText('loinc:30525-0');
  for (const cohortId of ['TIME-CHF', 'GISSI-HF']) {
    await expect(standardCodeGroup.getByText(`${cohortId}:`, {exact: true}).locator('..')).toContainText('age');
  }
  await checkpoint(page, '03-metadata-filters.png');

  await page.getByRole('button', {name: '✕ Clear'}).click();
  await timeCard.getByRole('button', {name: /Analyses & Insights/}).click();
  await expect(timeCard.locator('.stat').filter({hasText: 'Total Variables'}).locator('.stat-value')).toHaveText('35');
  await timeCard.getByRole('button', {name: 'Variance Ranking (numeric vars)'}).click();
  const edaRanking = timeCard.getByTestId('eda-cv-ranking');
  await expect(edaRanking.getByRole('heading', {name: 'Variables Ranked by Coefficient of Variation'})).toBeVisible();
  await edaRanking.getByTestId('eda-variable-row').first().click();
  const edaDetail = page.getByTestId('eda-variable-detail');
  await expect(edaDetail).toBeVisible();
  const edaOriginalGraph = edaDetail.getByTestId('eda-original-graph');
  await expect(edaOriginalGraph).toBeVisible();
  await expect
    .poll(() => edaOriginalGraph.evaluate((image: HTMLImageElement) => image.naturalWidth))
    .toBeGreaterThan(0);
  await edaDetail.getByRole('button', {name: '✕'}).click();

  await timeCard.getByRole('button', {name: 'Variables List'}).click();
  await page.getByTestId('variable-graph-TIME-CHF-age').click();
  const ageDistribution = page.getByRole('img', {name: 'age distribution graph'});
  const variableGraphModal = page.locator('.modal-box').filter({has: ageDistribution});
  await expect(ageDistribution).toBeVisible();
  await expect.poll(() => ageDistribution.evaluate((image: HTMLImageElement) => image.naturalWidth)).toBeGreaterThan(0);
  await variableGraphModal.getByRole('button', {name: 'Close', exact: true}).click();
  const ageConceptButton = page.getByTestId('concept-map-TIME-CHF-age');
  const ageConceptResponsePromise = page.waitForResponse(
    response => response.url().includes('/api/search-concepts?') && response.url().includes('query=age')
  );
  await ageConceptButton.click();
  const ageConceptDialog = page.locator('#autocomplete_concept_modal_TIME-CHF_age');
  expect((await ageConceptResponsePromise).status()).toBe(200);
  const ageInsertPromise = page.waitForResponse(
    response => response.url() === `${apiUrl}/insert-triples` && response.request().method() === 'POST'
  );
  await ageConceptDialog.getByRole('row').filter({hasText: 'loinc:30525-0'}).click();
  expect((await ageInsertPromise).status()).toBe(200);

  await page.getByTestId('variable-details-TIME-CHF-gender').click();
  const genderDetails = page.locator('#source_modal_TIME-CHF_gender');
  await expect(genderDetails).toBeVisible();
  const femaleConceptButton = page.getByTestId('concept-map-TIME-CHF-gender-category-0');
  const femaleSearchPromise = page.waitForResponse(
    response => response.url().includes('/api/search-concepts?') && response.url().includes('query=Female')
  );
  await femaleConceptButton.click();
  const femaleConceptDialog = page.locator('#autocomplete_concept_modal_TIME-CHF_gender-category-0');
  expect((await femaleSearchPromise).status()).toBe(200);
  const femaleInsertPromise = page.waitForResponse(
    response => response.url() === `${apiUrl}/insert-triples` && response.request().method() === 'POST'
  );
  await femaleConceptDialog.getByRole('row').filter({hasText: 'snomedct:248152002'}).click();
  expect((await femaleInsertPromise).status()).toBe(200);
  await expect(femaleConceptButton).toContainText('snomedct:248152002');
  await checkpoint(page, '04-manual-concept-mapping.png');

  await expect
    .poll(async () => (await currentMetadata(page))['TIME-CHF'].variables.age.mapped_id)
    .toBe('loinc:30525-0');
  await expect
    .poll(async () => (await currentMetadata(page))['TIME-CHF'].variables.gender.categories[0].mapped_id)
    .toBe('snomedct:248152002');
  await page.keyboard.press('Escape');
  await expect(genderDetails).not.toBeVisible();

  const originalTimeDictionary = readFileSync(dictionaryPath('TIME-CHF'), 'utf8');
  const replacementTimeDictionary = originalTimeDictionary.replace(
    '\nage,age,FLOAT,',
    '\nage,age at enrollment,FLOAT,'
  );
  expect(replacementTimeDictionary).not.toBe(originalTimeDictionary);
  const replacementDictionaryPath = testInfo.outputPath('TIME-CHF_changed-label_datadictionary.csv');
  mkdirSync(path.dirname(replacementDictionaryPath), {recursive: true});
  writeFileSync(replacementDictionaryPath, replacementTimeDictionary, 'utf8');
  const replacementDictionaryHash = sha256(Buffer.from(replacementTimeDictionary));

  await page.goto(`${browserUrl}/upload`);
  await page.locator('#upload-cohort-select').selectOption('TIME-CHF');
  await expect(page.getByText(/Metadata already exists for cohort/)).toBeVisible();
  await page.locator('#metadata-file').setInputFiles(replacementDictionaryPath);
  await page.locator('#validate-dictionary').click();
  await expect(page.locator('#validation-results')).toContainText('Validation successful');
  let persistedMetadata = await currentMetadata(page);
  expect(persistedMetadata['TIME-CHF'].variables.age.var_label).toBe('age');
  expect(persistedMetadata['TIME-CHF'].variables.age.mapped_id).toBe('loinc:30525-0');
  expect(persistedMetadata['TIME-CHF'].variables.gender.categories[0].mapped_id).toBe('snomedct:248152002');
  await page.locator('#validation-results').getByRole('button', {name: 'Got it'}).click();
  const replaceResponsePromise = page.waitForResponse(
    response => response.url() === `${apiUrl}/upload-cohort` && response.request().method() === 'POST'
  );
  await page.locator('#upload-dictionary').click();
  expect((await replaceResponsePromise).status()).toBe(200);
  persistedMetadata = await currentMetadata(page);
  expect(persistedMetadata['TIME-CHF'].variables.age.var_label).toBe('age at enrollment');
  expect(persistedMetadata['TIME-CHF'].variables.age.mapped_id).toBe('loinc:30525-0');
  expect(persistedMetadata['TIME-CHF'].variables.gender.categories[0].mapped_id).toBe('snomedct:248152002');

  const invalidTimeDictionary = replacementTimeDictionary.replace(
    'VARIABLENAME,VARIABLELABEL,VARTYPE,',
    'VARIABLENAME,VARIABLELABEL,NOT_A_TYPE,'
  );
  expect(invalidTimeDictionary).not.toBe(replacementTimeDictionary);
  const invalidDictionaryPath = testInfo.outputPath('TIME-CHF_invalid_datadictionary.csv');
  writeFileSync(invalidDictionaryPath, invalidTimeDictionary, 'utf8');
  await page.goto(`${browserUrl}/upload`);
  await page.locator('#upload-cohort-select').selectOption('TIME-CHF');
  await page.locator('#metadata-file').setInputFiles(invalidDictionaryPath);
  const invalidValidationResponse = page.waitForResponse(
    response => response.url() === `${apiUrl}/validate-cohort-dictionary` && response.request().method() === 'POST'
  );
  allowInvalidDictionaryValidation = true;
  await page.locator('#validate-dictionary').click();
  expect((await invalidValidationResponse).status()).toBe(422);
  allowInvalidDictionaryValidation = false;
  await expect(page.locator('#validation-results')).not.toContainText('Validation successful');
  await expect(page.locator('#validation-results')).toContainText(/VARTYPE|validation/i);
  persistedMetadata = await currentMetadata(page);
  expect(persistedMetadata['TIME-CHF'].variables.age.var_label).toBe('age at enrollment');
  expect(persistedMetadata['TIME-CHF'].variables.age.mapped_id).toBe('loinc:30525-0');
  expect(persistedMetadata['TIME-CHF'].variables.gender.categories[0].mapped_id).toBe('snomedct:248152002');
  expect(approvedLocalFailures).toEqual(['422 POST /validate-cohort-dictionary']);

  await page.goto(`${browserUrl}/mapping`);
  const sourceControl = page.getByTestId('mapping-source').locator('..');
  await sourceControl.getByText('TIME-CHF', {exact: true}).click();
  await page.getByTestId('mapping-target-GISSI-HF').check();
  const mappingDownloadPromise = page.waitForEvent('download');
  const mappingResponsePromise = page.waitForResponse(
    response => response.url() === `${apiUrl}/api/generate-mapping` && response.request().method() === 'POST'
  );
  await page.getByTestId('generate-mapping').click();
  expect((await mappingResponsePromise).status()).toBe(200);
  const mappingDownload = await mappingDownloadPromise;
  expect(mappingDownload.suggestedFilename()).toMatch(/time-chf.*gissi-hf.*\.csv/i);
  expect(sha256(await downloadBuffer(mappingDownload))).toBe(manifest.mapping_source.sha256);
  const mappingPreview = page.getByTestId('mapping-preview');
  await expect(mappingPreview).toContainText('Mapping Preview', {timeout: 30_000});
  await expect(mappingPreview).toContainText('Mappings per target: GISSI-HF (35)');
  await page.getByTestId('mapping-view-table').click();
  await expect(mappingPreview.getByRole('columnheader', {name: 'source variable'})).toBeVisible();
  await checkpoint(page, '05-generated-mapping-table.png');

  const mappingActivity = await responseJson(await page.request.get(`${apiUrl}/api/mapping-activity-log`));
  expect(mappingActivity.entries.some((entry: any) => entry.event === 'run_completed')).toBe(true);
  await page.getByTestId('mapping-view-graph').click();
  await expect(mappingPreview.locator('svg')).toBeVisible();
  await expect(mappingPreview).toContainText(/35 src · 35 tgt · 35 edges/);
  await checkpoint(page, '06-generated-mapping-graph.png');

  await page.getByRole('button', {name: 'show cached pairs'}).click();
  await expect(page.getByText('time-chf → gissi-hf', {exact: true})).toBeVisible();
  await page.getByRole('button', {name: '✕'}).click();

  await page.goto(`${browserUrl}/cohorts`);
  await addCohortToDcr(page, 'TIME-CHF', 1);
  await addCohortToDcr(page, 'GISSI-HF', 2);
  await page.getByTestId('dcr-launcher').click();
  await expect(page.getByTestId('dcr-local-simulation-warning')).toContainText(
    'does not provide a confidential-computing or production security boundary'
  );
  await expect(page.getByTestId('dcr-wizard-panel-name')).toBeVisible();
  await page.getByTestId('dcr-name-edit').click();
  await page.getByTestId('dcr-name-input').fill(roomName);
  await page.getByTestId('dcr-name-input').press('Enter');
  await expect(page.getByTestId('dcr-name-display')).toContainText(roomName);

  await page.getByTestId('dcr-wizard-next').click();
  await expect(page.getByTestId('dcr-wizard-panel-participants')).toBeVisible();
  await page.getByTestId('dcr-participants-open').click();
  const participantsModal = page.getByTestId('dcr-participants-modal');
  await expect(participantsModal).toContainText(adminEmail);
  const ownerToggle = participantsModal.getByTestId('dcr-participant-owner-toggle').first();
  await ownerToggle.check();
  await participantsModal.getByTestId('dcr-participant-analyst-input').fill(analystEmail);
  await participantsModal.getByTestId('dcr-participant-analyst-add').click();
  await expect(participantsModal).toContainText(analystEmail);
  await participantsModal.getByRole('button', {name: 'Done'}).click();

  await page.getByTestId('dcr-wizard-next').click();
  await page.getByTestId('dcr-research-question-input').fill(researchQuestion);
  await expect(page.getByTestId('dcr-research-question-input')).toHaveValue(researchQuestion);

  await page.getByTestId('dcr-wizard-next').click();
  const samplesPanel = page.getByTestId('dcr-wizard-panel-data-samples');
  await expect(samplesPanel).toContainText('Synthetic Samples');
  await expect(samplesPanel.getByTestId('dcr-sample-airlock')).toHaveCount(0);
  for (const cohortId of ['GISSI-HF', 'TIME-CHF']) {
    const sampleCard = samplesPanel.locator('.rounded-lg').filter({hasText: cohortId}).first();
    await sampleCard.getByTestId('dcr-sample-none').click();
    await sampleCard.getByTestId('dcr-sample-shuffled').click();
    await expect(sampleCard.getByTestId('dcr-sample-shuffled')).toHaveClass(/btn-primary/);
  }

  await page.getByTestId('dcr-wizard-next').click();
  const mappingPanel = page.getByTestId('dcr-wizard-panel-mapping');
  const mappingToggle = mappingPanel.getByTestId('dcr-mapping-toggle');
  await expect(mappingToggle).toHaveCount(1);
  await mappingToggle.check();
  await mappingPanel.getByTestId('dcr-mapping-upload-slot').check();

  await page.getByTestId('dcr-wizard-next').click();
  const reviewPanel = page.getByTestId('dcr-wizard-panel-review');
  await expect(reviewPanel).toContainText(roomName);
  await expect(reviewPanel).toContainText(adminEmail);
  await expect(reviewPanel).toContainText(analystEmail);
  await expect(reviewPanel).toContainText(researchQuestion);
  await expect(reviewPanel).toContainText('GISSI-HF, TIME-CHF');
  await expect(reviewPanel).toContainText('Shuffled Samples: GISSI-HF, TIME-CHF');
  await expect(reviewPanel).toContainText(/TIME-CHF.*GISSI-HF.*Upload slot/i);
  await checkpoint(page, '07-dcr-wizard-review.png');

  const previewResponsePromise = page.waitForResponse(
    response => response.url() === `${apiUrl}/get-compute-dcr-definition` && response.request().method() === 'POST'
  );
  const previewDownloadPromise = page.waitForEvent('download');
  await page.getByTestId('dcr-preview-download').click();
  const previewResponse = await previewResponsePromise;
  expect(previewResponse.status()).toBe(200);
  const previewDownload = await previewDownloadPromise;
  expect(previewDownload.suggestedFilename()).toBe('dcr_config_with_samples.zip');
  const previewPath = await previewDownload.path();
  expect(previewPath).not.toBeNull();
  const previewBytes = await downloadBuffer(previewDownload);
  expect(previewBytes.subarray(0, 2).toString()).toBe('PK');
  expect(sha256(previewBytes)).toBe(sha256(Buffer.from(await previewResponse.body())));
  const definitionHash = sha256(previewBytes);
  const persistedDefinition = path.join(evidenceDir, 'dcr-definition.zip');
  copyFileSync(previewPath as string, persistedDefinition);
  writeFileSync(path.join(evidenceDir, 'dcr-definition.sha256'), `${definitionHash}  dcr-definition.zip\n`, 'utf8');
  const previewPayload = previewResponse.request().postDataJSON();
  const previewReplay = await page.request.post(`${apiUrl}/get-compute-dcr-definition`, {data: previewPayload});
  expect(previewReplay.status()).toBe(200);
  expect(sha256(Buffer.from(await previewReplay.body()))).toBe(sha256(previewBytes));

  const expectedMembers = [
    'dcr_config.json',
    'fixture-provenance.json',
    'mapping_files/time-chf_gissi-hf_full.csv',
    'metadata_dictionaries/GISSI-HF_datadictionary.csv',
    'metadata_dictionaries/TIME-CHF_datadictionary.csv',
    'shuffled_samples/GISSI-HF_shuffled_sample.csv',
    'shuffled_samples/TIME-CHF_shuffled_sample.csv'
  ];
  expect(zipMembers(previewPath as string)).toEqual(expectedMembers);
  const definition = zipJson(previewPath as string, 'dcr_config.json').dataScienceDataRoom;
  expect(definition).toMatchObject({
    provider: 'aadcrv2',
    local_simulation: true,
    confidential_boundary: false,
    synthetic_demo: true,
    name: `${roomName} - created by ${adminEmail}`,
    cohorts: ['GISSI-HF', 'TIME-CHF']
  });
  const expectedDataNodeNames = [
    'GISSI-HF',
    'GISSI-HF_metadata_dictionary',
    'GISSI-HF_shuffled_sample',
    'TIME-CHF',
    'TIME-CHF_metadata_dictionary',
    'TIME-CHF_shuffled_sample',
    'time-chf_gissi-hf_mapping',
    'CrossStudyMappings'
  ];
  expect(definition.data_nodes.map((node: any) => node.name)).toEqual(expectedDataNodeNames);
  expect(definition.computation_nodes).toHaveLength(2);
  expect(definition.participants.map((participant: any) => participant.email)).toEqual([adminEmail, analystEmail]);
  const provenance = zipJson(previewPath as string, 'fixture-provenance.json');
  expect(provenance.provider).toBe('aadcrv2');
  expect(provenance.synthetic_fixture).toBe(true);
  expect(provenance.files).toHaveLength(5);
  const expectedDefinitionAssetHashes: Record<string, string> = {
    'mapping_files/time-chf_gissi-hf_full.csv': manifest.mapping_source.sha256,
    'metadata_dictionaries/GISSI-HF_datadictionary.csv':
      manifest.files['cohorts/GISSI-HF/GISSI-HF_datadictionary.csv'].sha256,
    'metadata_dictionaries/TIME-CHF_datadictionary.csv': replacementDictionaryHash,
    'shuffled_samples/GISSI-HF_shuffled_sample.csv': manifest.files['dcr_output_GISSI-HF/shuffled_sample.csv'].sha256,
    'shuffled_samples/TIME-CHF_shuffled_sample.csv': manifest.files['dcr_output_TIME-CHF/shuffled_sample.csv'].sha256
  };
  const provenanceByPath = Object.fromEntries(
    provenance.files.map((file: any) => [file.archive_path, file.sha256])
  ) as Record<string, string>;
  expect(provenanceByPath).toEqual(expectedDefinitionAssetHashes);
  for (const [archivePath, expectedHash] of Object.entries(expectedDefinitionAssetHashes)) {
    expect(zipMemberSha256(previewPath as string, archivePath)).toBe(expectedHash);
  }

  const createResponsePromise = page.waitForResponse(
    response => response.url() === `${apiUrl}/create-live-compute-dcr` && response.request().method() === 'POST'
  );
  await page.getByTestId('dcr-create').click();
  const createResponse = await createResponsePromise;
  expect(createResponse.status()).toBe(200);
  const created = await createResponse.json();
  await expect(page.getByTestId('dcr-wizard-success')).toBeVisible({timeout: 90_000});
  await expect(page.getByTestId('dcr-created-room-id')).toContainText(created.dcr_id);
  await expect(page.getByTestId('dcr-wizard-success')).toContainText(created.dcr_title);
  await expect(page.getByTestId('dcr-created-room-link')).toContainText('Open in My DCRs');
  expect(created.dcr_title).toBe(`${roomName} - created by ${adminEmail}`);
  await checkpoint(page, '08-dcr-created.png');

  const roomHref = await page.getByTestId('dcr-created-room-link').getAttribute('href');
  expect(roomHref).toBeTruthy();
  await page.goto(new URL(roomHref as string, browserUrl).toString());
  let roomCard = page.getByTestId('dcr-room-card');
  await expect(roomCard).toHaveCount(1);
  await expect(roomCard).toContainText(created.dcr_title);
  await expect(roomCard).toContainText(adminEmail);
  await expect(roomCard).toContainText(analystEmail);
  await expect(roomCard).toContainText('CrossStudyMappings');
  await expect(roomCard).toContainText('aggregate-summary-local-simulation');
  const provisioningRows = roomCard.getByTestId('dcr-provisioning-row');
  await expect(provisioningRows).toHaveCount(7);
  for (const cohortId of ['GISSI-HF', 'TIME-CHF']) {
    const rawProvision = provisioningRows.filter({
      has: page.locator('span').filter({hasText: new RegExp(`^${cohortId}$`)})
    });
    await expect(rawProvision).toHaveCount(1);
    await expect(rawProvision.locator('span').nth(0)).toHaveText(cohortId);
    await expect(rawProvision.locator('span').nth(1)).toHaveText(`→ ${cohortId}`);
    await expect(rawProvision.locator('span').nth(2)).toHaveText('provisioned');
  }

  const refreshResponsePromise = page.waitForResponse(
    response => response.url() === `${apiUrl}/my-dcrs/refresh` && response.request().method() === 'POST'
  );
  await page.getByTestId('my-dcrs-refresh').click();
  expect((await refreshResponsePromise).status()).toBe(200);
  await expect(roomCard).toContainText(created.dcr_title);
  const rooms = await responseJson(await page.request.get(`${apiUrl}/my-dcrs`));
  expect(rooms.dcrs).toHaveLength(1);
  expect(rooms.dcrs[0].id).toBe(created.dcr_id);
  expect(rooms.dcrs[0].title).toBe(created.dcr_title);
  expect(rooms.dcrs[0].participants).toHaveLength(2);
  expect(rooms.dcrs[0].nodes.map((node: any) => node.name).sort()).toEqual(
    [...expectedDataNodeNames, 'aggregate-summary-local-simulation', 'metadata-preview-local-simulation'].sort()
  );
  await checkpoint(page, '09-my-dcrs.png');

  await roomCard.getByTestId('dcr-room-audit-fetch').click();
  const auditEvents = roomCard.getByTestId('dcr-room-audit-events');
  await expect(auditEvents).toBeVisible();
  expect(await auditEvents.locator('tbody tr').count()).toBeGreaterThanOrEqual(5);
  await expect(auditEvents).toContainText('Create data clean room');
  await expect(auditEvents).toContainText('Create merge request');
  await expect(auditEvents).toContainText('Provision dataset');
  await checkpoint(page, '10-audit-log.png');

  const resultDownloadPromise = page.waitForEvent('download');
  await roomCard.getByTestId('dcr-result-run').click();
  const resultDownload = await resultDownloadPromise;
  await expect(roomCard.getByTestId('dcr-result-ready')).toContainText('aggregate-result.zip');
  expect(resultDownload.suggestedFilename()).toBe('aggregate-result.zip');
  expect(sha256(await downloadBuffer(resultDownload))).toBe(expectedResultHash);
  await checkpoint(page, '11-aggregate-result.png');

  await roomCard.getByTestId('dcr-room-audit-fetch').click();
  await expect(auditEvents.locator('tbody tr')).not.toHaveCount(0);
  await expect(auditEvents).toContainText('Run computation');

  await page.reload();
  roomCard = page.getByTestId('dcr-room-card');
  await expect(roomCard).toHaveCount(1);
  await expect(roomCard).toContainText(created.dcr_title);

  await page.goto(`${browserUrl}/cohorts`);
  await openCohort(page, 'TIME-CHF');
  await page.getByTestId('cohort-TIME-CHF').getByRole('button', {name: 'Variables List'}).click();
  await expect(page.getByTestId('variable-TIME-CHF-age')).toContainText('age at enrollment');
  await expect(page.getByTestId('concept-map-TIME-CHF-age')).toContainText('loinc:30525-0');
  await page.getByTestId('variable-details-TIME-CHF-gender').click();
  await expect(page.getByTestId('concept-map-TIME-CHF-gender-category-0')).toContainText('snomedct:248152002');

  await page.goto(`${browserUrl}/mapping`);
  await page.getByRole('button', {name: 'show cached pairs'}).click();
  const cachedPair = page.getByRole('row').filter({hasText: 'time-chf → gissi-hf'});
  await expect(cachedPair).toBeVisible();
  await cachedPair.getByRole('button', {name: 'Show table'}).click();
  await expect(page.getByTestId('mapping-preview')).toContainText('Mapping Preview');

  await page.goto(`${browserUrl}/dcrs`);
  await expect(page.getByTestId('dcr-room-card')).toHaveCount(1);
  await expect(page.getByTestId('dcr-room-card')).toContainText(created.dcr_title);

  expect(externalRequests).toEqual([]);
  expect(externalWebSockets).toEqual([]);
  expect(pageErrors).toEqual([]);
  expect(consoleErrors).toEqual([]);
  expect(failedLocalResponses).toEqual([]);
  writeFileSync(
    path.join(evidenceDir, 'acceptance-details.json'),
    `${JSON.stringify(
      {
        console_errors: consoleErrors,
        approved_pre_auth_console_errors: preAuthConsoleErrors,
        approved_pre_auth_responses: preAuthFailedLocalResponses,
        approved_local_failures: approvedLocalFailures,
        definition_sha256: definitionHash,
        external_requests: externalRequests,
        external_websockets: externalWebSockets,
        failed_local_responses: failedLocalResponses,
        page_errors: pageErrors,
        result_sha256: expectedResultHash,
        room_id: created.dcr_id
      },
      null,
      2
    )}\n`,
    'utf8'
  );
});

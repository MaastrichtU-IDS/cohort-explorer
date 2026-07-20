import {createHash} from 'crypto';
import {mkdirSync, readFileSync, writeFileSync} from 'fs';
import path from 'path';

import {expect, test, type APIResponse, type Download, type Page, type Request} from '@playwright/test';

const browserUrl = process.env.DEMO_BROWSER_URL || 'http://localhost:3001';
const apiUrl = process.env.DEMO_API_URL || 'http://localhost:3000';
const aadcrUiUrl = process.env.DEMO_AADCR_UI_URL || 'http://localhost:3002';
const packDir = process.env.DEMO_BROWSER_PACK;
const evidenceDir = process.env.DEMO_BROWSER_EVIDENCE || path.resolve(__dirname, '../../artifacts/browser-demo');
const adminEmail = 'nikolas.molyndris@decentriq.ch';
const analystEmail = 'browser.analyst@example.com';
const roomName = 'Cohort Explorer Local Browser Acceptance';

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
  await expect(page.getByTestId('upload-dcr-step-title')).toHaveText('Step 2: Create Advanced Analytics DCR Handoff');
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
  const card = page.getByTestId(`cohort-card-${cohortId}`);
  await card.scrollIntoViewIfNeeded();
  if (!(await card.getByRole('button', {name: 'Variables List'}).isVisible())) {
    await card.locator('.collapse-title').click();
  }
  await expect(card.getByRole('button', {name: 'Variables List'})).toBeVisible();
}

async function addCohortToDcr(page: Page, cohortId: string, expectedCount: number): Promise<void> {
  await openCohort(page, cohortId);
  const card = page.getByTestId(`cohort-card-${cohortId}`);
  await card.getByRole('button', {name: 'Add to DCR'}).click();
  await expect(page.getByTestId('dcr-launcher')).toContainText(String(expectedCount), {timeout: 5_000});
}

test('complete metadata journey hands off to the real AADCR v2 development workflow', async ({page}, testInfo) => {
  const expectedPreAuthConsoleError = 'Error fetching data in cache worker: Not authenticated';
  const expectedInvalidDictionaryConsoleError =
    'Failed to load resource: the server responded with a status of 422 (Unprocessable Entity)';
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
    const card = page.getByTestId(`cohort-card-${cohortId}`);
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
  await expect(page.getByTestId('cohort-card-TIME-CHF')).toBeVisible();
  await expect(page.getByTestId('cohort-card-GISSI-HF')).toHaveCount(0);
  await prospectiveStudy.uncheck();

  const providerFilter = page.getByTestId('metadata-filter-institution');
  const timeProvider = providerFilter
    .getByText('Synthetic iCARE4CVD Demo Consortium - TIME-CHF Site (1)', {exact: true})
    .locator('..')
    .getByRole('checkbox');
  await timeProvider.check();
  await expect(page.getByTestId('cohort-card-TIME-CHF')).toBeVisible();
  await expect(page.getByTestId('cohort-card-GISSI-HF')).toHaveCount(0);
  await timeProvider.uncheck();

  const cohortSearch = page.getByTestId('cohort-search');
  await page.getByRole('button', {name: /Cohorts Metadata/}).click();
  await page.getByRole('button', {name: /OR Search/}).click();
  await cohortSearch.fill('TIME GISSI');
  await expect(page.getByText(/Found\s+2\s+cohorts with matches/)).toBeVisible();
  await expect(page.getByTestId('cohort-card-TIME-CHF')).toBeVisible();
  await expect(page.getByTestId('cohort-card-GISSI-HF')).toBeVisible();
  await page.getByRole('button', {name: /AND Search/}).click();
  await expect(page.getByText(/No matches found for/)).toBeVisible();
  await expect(page.getByText('0/2 cohorts', {exact: true})).toBeVisible();
  await page.getByRole('button', {name: /Exact Phrase/}).click();
  await cohortSearch.fill('Cohort study');
  await expect(page.getByText(/Found\s+2\s+cohorts with matches/)).toBeVisible();
  await expect(page.getByText(/study design/i).first()).toBeVisible();
  await cohortSearch.fill('Synthetic iCARE4CVD Demo Consortium');
  await expect(page.getByText(/Found\s+2\s+cohorts with matches/)).toBeVisible();
  await expect(page.getByText(/institution/i).first()).toBeVisible();
  await page.getByRole('button', {name: '✕ Clear'}).click();

  await openCohort(page, 'TIME-CHF');
  const timeCard = page.getByTestId('cohort-card-TIME-CHF');
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
  await openCohort(page, 'TIME-CHF');
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
  await genderDetails.getByTestId('variable-details-close-TIME-CHF-gender').click();
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
  const mappingDownloadPromise = page.waitForEvent('download', {timeout: 90_000});
  const mappingResponsePromise = page.waitForResponse(
    response => response.url() === `${apiUrl}/api/generate-mapping` && response.request().method() === 'POST',
    {timeout: 90_000}
  );
  await page.getByTestId('generate-mapping').click();
  expect((await mappingResponsePromise).status()).toBe(200);
  const mappingDownload = await mappingDownloadPromise;
  expect(mappingDownload.suggestedFilename()).toBe('time-chf_gissi-hf_fixture.json');
  const mappingDownloadBytes = await downloadBuffer(mappingDownload);
  const generatedMapping = JSON.parse(mappingDownloadBytes.toString('utf8'));
  const generatedMappingRows = Object.entries(generatedMapping).flatMap(([source, value]: [string, any]) =>
    value.mappings.map((mapping: any) => ({
      source,
      target: mapping.target,
      sourceStudy: mapping.source_study,
      targetStudy: mapping.target_study,
      relation: mapping.mapping_relation,
      status: mapping.harmonization_status
    }))
  );
  expect(generatedMappingRows).toHaveLength(1289);
  expect(generatedMappingRows.every(row => row.sourceStudy === 'time-chf')).toBe(true);
  expect(generatedMappingRows.every(row => row.targetStudy === 'gissi-hf')).toBe(true);
  expect(generatedMappingRows.every(row => row.status === 'pending')).toBe(true);
  const generatedMappingPairs = new Set(generatedMappingRows.map(row => `${row.source}->${row.target}`));
  for (const row of manifest.selected_mapping_rows) {
    expect(generatedMappingPairs.has(`${row.source}->${row.target}`)).toBe(true);
  }
  const mappingPreview = page.getByTestId('mapping-preview');
  await expect(mappingPreview).toContainText('Mapping Preview', {timeout: 30_000});
  await expect(mappingPreview).toContainText('Mappings per target: gissi-hf (1289)');
  await page.getByTestId('mapping-view-table').click();
  await expect(mappingPreview.getByRole('columnheader', {name: 'source variable'})).toBeVisible();
  await checkpoint(page, '05-generated-mapping-table.png');

  const mappingActivity = await responseJson(await page.request.get(`${apiUrl}/api/mapping-activity-log`));
  expect(mappingActivity.entries.some((entry: any) => entry.event === 'run_completed')).toBe(true);
  const fixtureActivity = mappingActivity.entries.find((entry: any) => entry.event === 'fixture_materialized');
  expect(fixtureActivity).toBeDefined();
  expect(fixtureActivity.ctx.total_mappings).toBe(1289);
  expect(fixtureActivity.ctx.output_sha256[mappingDownload.suggestedFilename()]).toBe(sha256(mappingDownloadBytes));
  await page.getByTestId('mapping-view-graph').click();
  await expect(mappingPreview.locator('svg')).toBeVisible();
  await expect(mappingPreview).toContainText(/379 source · 397 target · 1289 edges/);
  await checkpoint(page, '06-generated-mapping-graph.png');

  await page.getByRole('button', {name: 'show cached pairs'}).click();
  await expect(page.getByText('time-chf → gissi-hf', {exact: true})).toBeVisible();
  await page.getByTestId('mapping-cache-close').click();

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
  const mappingPanel = page.getByTestId('dcr-wizard-panel-mapping');
  await expect(mappingPanel).toBeVisible();
  const mappingToggle = mappingPanel.getByTestId('dcr-mapping-toggle');
  await expect(mappingToggle).toHaveCount(1);
  await mappingToggle.check();
  await mappingPanel.getByTestId('dcr-mapping-upload-slot').check();

  await page.getByTestId('dcr-wizard-next').click();
  const reviewPanel = page.getByTestId('dcr-wizard-panel-review');
  await expect(reviewPanel).toContainText(roomName);
  await expect(reviewPanel).toContainText('GISSI-HF, TIME-CHF');
  await expect(reviewPanel).toContainText(/TIME-CHF.*GISSI-HF.*Upload slot/i);
  await expect(page.getByTestId('dcr-handoff-boundary')).toContainText(
    'Participants, synthetic dataset upload and provisioning, permissions, computations, change requests, results, and audit history remain in the Advanced Analytics DCR.'
  );
  await expect(reviewPanel.getByTestId('dcr-preview-download')).toHaveCount(0);
  await checkpoint(page, '07-dcr-handoff-review.png');

  const expectedDataNodeNames = [
    'CrossStudyMappings',
    'GISSI-HF',
    'GISSI-HF_metadata_dictionary',
    'GISSI-HF_shuffled_sample',
    'TIME-CHF',
    'TIME-CHF_metadata_dictionary',
    'TIME-CHF_shuffled_sample',
    'time-chf_gissi-hf_mapping'
  ];

  const createResponsePromise = page.waitForResponse(
    response => response.url() === `${apiUrl}/create-live-compute-dcr` && response.request().method() === 'POST'
  );
  await page.getByTestId('dcr-create').click();
  const createResponse = await createResponsePromise;
  expect(createResponse.status()).toBe(200);
  const created = await createResponse.json();
  expect(created).toMatchObject({
    environment: 'DEV',
    handoff_mode: 'bootstrap',
    mapping_upload_results: {},
    mapping_uploads_successful: 0,
    metadata_upload_results: {},
    metadata_uploads_successful: 0,
    participants: {},
    provider: 'aadcrv2',
    row_upload_results: {},
    row_uploads_successful: 0,
    shuffled_upload_results: {},
    shuffled_uploads_successful: 0
  });
  expect(created).not.toHaveProperty('aggregate_computation_node_id');
  expect(created).not.toHaveProperty('merge_request_id');
  expect(Object.keys(created.data_node_ids).sort()).toEqual(expectedDataNodeNames);
  expect(created.dcr_url).toBe(`${aadcrUiUrl}/aadcrv2/dcr/${created.dcr_id}`);
  await expect(page.getByTestId('dcr-wizard-success')).toBeVisible({timeout: 90_000});
  await expect(page.getByTestId('dcr-created-room-id')).toContainText(created.dcr_id);
  await expect(page.getByTestId('dcr-wizard-success')).toContainText(created.dcr_title);
  await expect(page.getByTestId('dcr-created-room-link')).toContainText('Open Advanced Analytics DCR');
  await expect(page.getByTestId('dcr-bootstrap-next-steps')).toContainText('8 metadata-derived data slots');
  expect(created.dcr_title).toBe(`${roomName} - created by ${adminEmail}`);
  await expect(page.getByRole('link', {name: 'My DCRs'})).toHaveCount(0);
  await checkpoint(page, '08-dcr-handoff-created.png');

  const roomHref = await page.getByTestId('dcr-created-room-link').getAttribute('href');
  expect(roomHref).toBe(created.dcr_url);
  await page.goto(created.dcr_url);
  await expect(page.getByRole('heading', {name: created.dcr_title})).toBeVisible({timeout: 30_000});
  await expect(page.getByRole('tab', {name: 'Production'})).toBeVisible();
  await expect(page.getByRole('tab', {name: 'Development'})).toBeVisible();
  await expect(page.getByRole('tab', {name: 'Change requests'})).toBeVisible();
  await expect(page.getByRole('tab', {name: 'Audit log'})).toBeVisible();
  await checkpoint(page, '09-aadcr-original-production.png');

  await page.getByRole('tab', {name: 'Development'}).click();
  await expect(page.getByRole('button', {name: 'Add computation node'})).toBeVisible();
  for (const [nodeName, nodeId] of Object.entries(created.data_node_ids) as Array<[string, string]>) {
    await expect(page.getByTestId(`aadcr-data-node-${nodeId}`)).toContainText(nodeName);
  }
  await expect(page.getByText('No participants added yet.')).toBeVisible();

  const timeNode = page.getByTestId(`aadcr-data-node-${created.data_node_ids['TIME-CHF']}`);
  const fileChooserPromise = page.waitForEvent('filechooser');
  await timeNode.getByRole('button', {name: 'Add dataset'}).click();
  const fileChooser = await fileChooserPromise;
  const provisionResponsePromise = page.waitForResponse(
    response =>
      response.url().endsWith(`/aadcr-api/dcr/${created.dcr_id}/provision-dataset`) &&
      response.request().method() === 'POST'
  );
  await fileChooser.setFiles(path.join(packDir, manifest.cohorts['TIME-CHF'].rows));
  const provisionResponse = await provisionResponsePromise;
  expect(provisionResponse.status()).toBe(200);
  await expect(timeNode).toContainText('TIME-CHF.csv', {timeout: 30_000});
  await checkpoint(page, '10-aadcr-synthetic-upload.png');

  const computationResponsePromise = page.waitForResponse(
    response =>
      response.url().endsWith(`/aadcr-api/dcr/${created.dcr_id}/dev/computation-nodes`) &&
      response.request().method() === 'POST'
  );
  await page.getByRole('button', {name: 'Add computation node'}).click();
  expect((await computationResponsePromise).status()).toBe(200);
  await expect(page.getByPlaceholder('Computation name')).toHaveValue('computation 1');
  await expect(page.getByText('Data dependencies', {exact: true})).toBeVisible();
  await checkpoint(page, '11-aadcr-computation-editor.png');
  await page.locator('.MuiModalClose-root:visible').click();

  await page.getByRole('button', {name: 'Add participant node'}).click();
  const participantDialog = page.getByRole('dialog').filter({hasText: 'Add participant'});
  await participantDialog.getByRole('textbox', {name: 'Email'}).fill(analystEmail);
  const participantResponsePromise = page.waitForResponse(
    response =>
      response.url().endsWith(`/aadcr-api/dcr/${created.dcr_id}/dev/participants`) &&
      response.request().method() === 'POST'
  );
  await participantDialog.getByRole('button', {name: 'Save'}).click();
  expect((await participantResponsePromise).status()).toBe(200);
  await expect(page.getByText(analystEmail, {exact: true})).toBeVisible();

  await page.getByRole('button', {name: 'Create change request'}).click();
  const changeRequestDialog = page.getByRole('dialog').filter({hasText: 'Create change request'});
  await changeRequestDialog.getByRole('textbox', {name: 'Title'}).fill('Synthetic cohort development changes');
  await changeRequestDialog
    .getByRole('textbox', {name: 'Description'})
    .fill('Upload and metadata-derived development nodes prepared through the local Cohort Explorer handoff.');
  const mergeResponsePromise = page.waitForResponse(
    response =>
      response.url().endsWith(`/aadcr-api/dcr/${created.dcr_id}/merge-requests/`) &&
      response.request().method() === 'POST'
  );
  await changeRequestDialog.getByRole('button', {name: 'Create change request'}).click();
  expect((await mergeResponsePromise).status()).toBe(201);
  await page.getByRole('tab', {name: 'Change requests'}).click();
  await expect(page.getByText('Synthetic cohort development changes', {exact: true})).toBeVisible();
  await checkpoint(page, '12-aadcr-change-request.png');

  await page.getByRole('tab', {name: 'Audit log'}).click();
  await expect(page.getByText(/Provision dataset/i).first()).toBeVisible();
  await expect(page.getByText(/Create merge request/i).first()).toBeVisible();
  await checkpoint(page, '13-aadcr-audit-log.png');

  const approvedConsoleErrors = consoleErrors.filter(message => message === expectedInvalidDictionaryConsoleError);
  const unexpectedConsoleErrors = consoleErrors.filter(message => message !== expectedInvalidDictionaryConsoleError);
  expect(externalRequests).toEqual([]);
  expect(externalWebSockets).toEqual([]);
  expect(pageErrors).toEqual([]);
  expect(approvedConsoleErrors).toEqual([expectedInvalidDictionaryConsoleError]);
  expect(unexpectedConsoleErrors).toEqual([]);
  expect(failedLocalResponses).toEqual([]);
  writeFileSync(
    path.join(evidenceDir, 'acceptance-details.json'),
    `${JSON.stringify(
      {
        console_errors: unexpectedConsoleErrors,
        approved_console_errors: approvedConsoleErrors,
        approved_pre_auth_console_errors: preAuthConsoleErrors,
        approved_pre_auth_responses: preAuthFailedLocalResponses,
        approved_local_failures: approvedLocalFailures,
        aadcr_handoff_url: created.dcr_url,
        data_node_count: Object.keys(created.data_node_ids).length,
        external_requests: externalRequests,
        external_websockets: externalWebSockets,
        failed_local_responses: failedLocalResponses,
        page_errors: pageErrors,
        room_id: created.dcr_id
      },
      null,
      2
    )}\n`,
    'utf8'
  );
});

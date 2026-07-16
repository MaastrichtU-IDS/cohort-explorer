import {describe, expect, it} from 'vitest';

import checkAnalysisFolder from '../src/pages/api/check-analysis-folder/[cohortId].js';
import cohortEdaOutput from '../src/pages/api/cohort-eda-output/[cohortName].js';
import variableGraph from '../src/pages/api/variable-graph/[cohortId]/[variableName].js';

const responseRecorder = () => {
  const state: {status?: number; body?: unknown} = {};
  const response = {
    status(code: number) {
      state.status = code;
      return response;
    },
    json(body: unknown) {
      state.body = body;
      return response;
    },
    end() {
      return response;
    },
    send(body: unknown) {
      state.body = body;
      return response;
    },
    setHeader() {
      return response;
    }
  };
  return {response, state};
};

describe('synthetic data API route containment', () => {
  it('rejects EDA traversal before resolving a file', () => {
    const {response, state} = responseRecorder();
    cohortEdaOutput({method: 'GET', query: {cohortName: '../../../../app/package'}}, response);
    expect(state.status).toBe(400);
    expect(state.body).toEqual({detail: 'Invalid cohortName'});
  });

  it('rejects analysis-folder traversal', () => {
    const {response, state} = responseRecorder();
    checkAnalysisFolder({method: 'GET', query: {cohortId: '../../etc'}}, response);
    expect(state.status).toBe(400);
    expect(state.body).toEqual({exists: false, error: 'Invalid cohortId'});
  });

  it('rejects traversal in either variable-graph segment', () => {
    for (const query of [
      {cohortId: '../outside', variableName: 'age'},
      {cohortId: 'TIME-CHF', variableName: '../../../secret'}
    ]) {
      const {response, state} = responseRecorder();
      variableGraph({method: 'GET', query}, response);
      expect(state.status).toBe(400);
      expect(state.body).toEqual({error: 'Invalid cohortId or variableName'});
    }
  });
});

import {describe, expect, it} from 'vitest';

import {isLatestMetadataResponse} from '@/utils/metadataRequest';

describe('metadata worker request sequencing', () => {
  it('accepts only the response for the most recently issued request', () => {
    expect(isLatestMetadataResponse(8, 9)).toBe(false);
    expect(isLatestMetadataResponse(9, 9)).toBe(true);
    expect(isLatestMetadataResponse(undefined, 9)).toBe(false);
  });
});

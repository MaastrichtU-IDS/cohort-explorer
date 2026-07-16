export const isLatestMetadataResponse = (
  responseRequestId: number | undefined,
  latestIssuedRequestId: number
): boolean => responseRequestId === latestIssuedRequestId;

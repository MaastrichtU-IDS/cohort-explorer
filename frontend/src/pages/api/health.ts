import type {NextApiRequest, NextApiResponse} from 'next';

type HealthResponse = {
  status: 'ok';
  service: 'frontend';
};

export default function handler(
  _request: NextApiRequest,
  response: NextApiResponse<HealthResponse>
) {
  response.status(200).json({status: 'ok', service: 'frontend'});
}

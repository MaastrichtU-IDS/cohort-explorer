// Stand-alone page for the no-code analysis wizard (the same wizard is also
// embedded in the "Create analysis DCR" modal behind the Flexible / No-code
// flexible wizard's switch link).
import React from 'react';
import NocodeWizard from '@/components/nocode/NocodeWizard';

export default function NocodeDcrPage() {
  return <NocodeWizard />;
}

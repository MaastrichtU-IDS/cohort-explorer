import {describe, expect, it, vi} from 'vitest';

import {abortOnPageExit} from '../pageExit';

describe('page-exit request cancellation', () => {
  it('aborts an in-flight request when pagehide fires', () => {
    const target = new EventTarget();
    const controller = new AbortController();
    const detach = abortOnPageExit(controller, target);

    target.dispatchEvent(new Event('pagehide'));

    expect(controller.signal.aborted).toBe(true);
    detach();
  });

  it('removes the page-exit listener during normal effect cleanup', () => {
    const target = new EventTarget();
    const controller = new AbortController();
    const detach = abortOnPageExit(controller, target);

    detach();
    target.dispatchEvent(new Event('pagehide'));

    expect(controller.signal.aborted).toBe(false);
  });

  it('restarts an aborted request only after a persisted page is restored', () => {
    const target = new EventTarget();
    const controller = new AbortController();
    const onPersistedRestore = vi.fn();
    const detach = abortOnPageExit(controller, target, onPersistedRestore);
    const regularShow = new Event('pageshow');
    const persistedShow = new Event('pageshow');
    Object.defineProperty(regularShow, 'persisted', {value: false});
    Object.defineProperty(persistedShow, 'persisted', {value: true});

    target.dispatchEvent(new Event('pagehide'));
    target.dispatchEvent(regularShow);
    target.dispatchEvent(persistedShow);

    expect(controller.signal.aborted).toBe(true);
    expect(onPersistedRestore).toHaveBeenCalledTimes(1);
    detach();
  });
});

// This file declares the JSX namespace to fix TypeScript errors with JSX elements
declare namespace JSX {
  interface IntrinsicElements {
    [elemName: string]: any;
  }
}

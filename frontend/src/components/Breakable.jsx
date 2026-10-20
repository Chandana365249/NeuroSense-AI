import { Fragment } from 'react';

/**
 * Renders an identifier (e.g. a model version) with optional line-break points after its natural separators
 * (_ - .), so it wraps between parts instead of in the middle of a word. Copying the text is unaffected.
 */
export default function Breakable({ text }) {
  if (typeof text !== 'string') return text ?? null;
  const parts = text.split(/(?<=[_\-.])/);
  return parts.map((part, i) => (
    // eslint-disable-next-line react/no-array-index-key
    <Fragment key={i}>{part}{i < parts.length - 1 ? <wbr /> : null}</Fragment>
  ));
}

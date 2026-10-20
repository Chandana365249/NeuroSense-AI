import PageHeader from '../components/PageHeader';
import { href } from '../router';

export default function NotFound({ path }) {
  return (
    <>
      <PageHeader title="Page not found" />
      <p className="muted">There is no page at <span className="mono">{path}</span>.</p>
      <p><a className="btn btn-secondary" href={href('/overview')}>Go to Overview</a></p>
    </>
  );
}

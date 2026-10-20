import AuthForm from './AuthForm';
import Card from './Card';
import LoadingBlock from './LoadingBlock';
import Notice from './Notice';

/**
 * Renders children only for a signed-in user (and, with `admin`, only for administrators).
 * This is a usability separation in the interface; the backend enforces its own authorization.
 */
export default function RequireAuth({ auth, admin = false, reason, children }) {
  if (auth.status === 'checking') return <LoadingBlock label="Checking your session…" />;
  if (auth.status === 'error') {
    return (
      <Notice tone="error" title="Could not verify your session">
        {auth.error || 'The service is unreachable.'} Please try again shortly.
      </Notice>
    );
  }
  if (!auth.user) {
    return (
      <Card title="Sign in required" titleId="signin-required" className="narrow">
        <p className="muted">{reason || 'Sign in to continue.'}</p>
        <AuthForm onAuth={auth.signIn} />
      </Card>
    );
  }
  if (admin && !auth.user.is_admin) {
    return <Notice tone="warning" title="Administrator access required">This page is only available to research administrators.</Notice>;
  }
  return children;
}

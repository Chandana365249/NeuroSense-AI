import { useEffect } from 'react';
import AuthForm from '../components/AuthForm';
import Card from '../components/Card';
import LoadingBlock from '../components/LoadingBlock';
import PageHeader from '../components/PageHeader';
import { navigate } from '../router';

/** Only internal routes are accepted as the post-sign-in destination. */
export function safeNext(next) {
  return next && /^\/[a-z0-9/_-]*$/i.test(next) && next !== '/sign-in' ? next : '/overview';
}

export default function SignIn({ auth, next }) {
  const target = safeNext(next);
  useEffect(() => {
    if (auth.user) navigate(target, { replace: true });
  }, [auth.user, target]);

  return (
    <>
      <PageHeader title="Sign in" subtitle="Sign in to analyze EEG recordings and see your history." />
      {auth.status === 'checking' || auth.user ? <LoadingBlock label="Signing you in…" /> : (
        <Card className="narrow"><AuthForm onAuth={auth.signIn} /></Card>
      )}
    </>
  );
}

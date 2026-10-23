/** Only internal routes are accepted as the post-sign-in destination. */
export function safeNext(next) {
  return next && /^\/[a-z0-9/_-]*$/i.test(next) && next !== '/sign-in' ? next : '/overview';
}

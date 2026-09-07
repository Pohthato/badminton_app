export type AuthAction = "profile" | "sign_in" | "configure_local_auth";

export function getAuthAction(isAuthenticated: boolean, hasOAuthConfig: boolean): AuthAction {
  if (isAuthenticated) return "profile";
  return hasOAuthConfig ? "sign_in" : "configure_local_auth";
}

export function getAuthButtonLabel(action: AuthAction) {
  if (action === "profile") return "Open profile menu";
  if (action === "sign_in") return "Sign in to analyze";
  return "Configure local sign-in";
}

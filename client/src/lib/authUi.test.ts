import { describe, expect, it } from "vitest";
import { getAuthAction, getAuthButtonLabel } from "./authUi";

describe("auth control state", () => {
  it("opens a profile menu for authenticated users", () => {
    const action = getAuthAction(true, false);
    expect(action).toBe("profile");
    expect(getAuthButtonLabel(action)).toBe("Open profile menu");
  });

  it("starts sign-in when hosted OAuth is configured", () => {
    const action = getAuthAction(false, true);
    expect(action).toBe("sign_in");
    expect(getAuthButtonLabel(action)).toBe("Sign in to analyze");
  });

  it("explains local configuration when OAuth is unavailable", () => {
    const action = getAuthAction(false, false);
    expect(action).toBe("configure_local_auth");
    expect(getAuthButtonLabel(action)).toBe("Configure local sign-in");
  });
});

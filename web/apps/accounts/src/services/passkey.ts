import { isDevBuild } from "@/next/env";
import { authenticatedRequestHeaders } from "@/next/http";
import log from "@/next/log";
import { ensure } from "@/utils/ensure";
import { nullToUndefined } from "@/utils/transform";
import { toB64URLSafeNoPadding } from "@ente/shared/crypto/internal/libsodium";
import HTTPService from "@ente/shared/network/HTTPService";
import { apiOrigin, getEndpoint } from "@ente/shared/network/api";
import { getToken } from "@ente/shared/storage/localStorage/helpers";
import _sodium from "libsodium-wrappers";

const ENDPOINT = getEndpoint();

/**
 * Variant of {@link authenticatedRequestHeaders} but for authenticated requests
 * made by the accounts app.
 *
 * We cannot use {@link authenticatedRequestHeaders} directly because the
 * accounts app does not save a full user and instead only saves the user's
 * token (and that token too is scoped to the accounts APIs).
 */
const accountsAuthenticatedRequestHeaders = (): Record<string, string> => {
    const token = getToken();
    if (!token) throw new Error("Missing accounts token");
    const headers: Record<string, string> = { "X-Auth-Token": token };
    const clientPackage = nullToUndefined(
        localStorage.getItem("clientPackage"),
    );
    if (clientPackage) headers["X-Client-Package"] = clientPackage;
    return headers;
};

export interface Passkey {
    id: string;
    userID: number;
    friendlyName: string;
    createdAt: number;
}

export const getPasskeys = async () => {
    const token = getToken();
    if (!token) return;
    const response = await HTTPService.get(
        `${ENDPOINT}/passkeys`,
        {},
        { "X-Auth-Token": token },
    );
    return await response.data;
};

export const renamePasskey = async (id: string, name: string) => {
    try {
        const token = getToken();
        if (!token) return;
        const response = await HTTPService.patch(
            `${ENDPOINT}/passkeys/${id}`,
            {},
            { friendlyName: name },
            { "X-Auth-Token": token },
        );
        return await response.data;
    } catch (e) {
        log.error("rename passkey failed", e);
        throw e;
    }
};

export const deletePasskey = async (id: string) => {
    try {
        const token = getToken();
        if (!token) return;
        const response = await HTTPService.delete(
            `${ENDPOINT}/passkeys/${id}`,
            {},
            {},
            { "X-Auth-Token": token },
        );
        return await response.data;
    } catch (e) {
        log.error("delete passkey failed", e);
        throw e;
    }
};

/**
 * Add a new passkey as the second factor to the user's account.
 *
 * @param name An arbitrary name that the user wishes to label this passkey with
 * (aka "friendly name").
 */
export const registerPasskey = async (name: string) => {
    const response: {
        options: {
            publicKey: PublicKeyCredentialCreationOptions;
        };
        sessionID: string;
    } = await getPasskeyRegistrationOptions();

    const options = response.options;

    // TODO-PK: The types don't match.
    options.publicKey.challenge = _sodium.from_base64(
        // eslint-disable-next-line @typescript-eslint/ban-ts-comment
        // @ts-ignore
        options.publicKey.challenge,
    );
    options.publicKey.user.id = _sodium.from_base64(
        // eslint-disable-next-line @typescript-eslint/ban-ts-comment
        // @ts-ignore
        options.publicKey.user.id,
    );

    // create new credential
    const credential = ensure(await navigator.credentials.create(options));

    await finishPasskeyRegistration(name, credential, response.sessionID);
};

export const getPasskeyRegistrationOptions = async () => {
    const url = `${apiOrigin()}/passkeys/registration/begin`;
    const res = await fetch(url, {
        headers: accountsAuthenticatedRequestHeaders(),
    });
    if (!res.ok) throw new Error(`Failed to fetch ${url}: HTTP ${res.status}`);
    return await res.json();
};

interface PasskeyRegistrationOptions {
    sessionID: string;
    options: {
        publicKey: PublicKeyCredentialCreationOptions;
    };
}

const finishPasskeyRegistration = async (
    friendlyName: string,
    credential: Credential,
    sessionID: string,
) => {
    const attestationObjectB64 = await toB64URLSafeNoPadding(
        // eslint-disable-next-line @typescript-eslint/ban-ts-comment
        // @ts-ignore
        new Uint8Array(credential.response.attestationObject),
    );
    const clientDataJSONB64 = await toB64URLSafeNoPadding(
        // eslint-disable-next-line @typescript-eslint/ban-ts-comment
        // @ts-ignore
        new Uint8Array(credential.response.clientDataJSON),
    );

    const token = ensure(getToken());

    const response = await HTTPService.post(
        `${ENDPOINT}/passkeys/registration/finish`,
        JSON.stringify({
            id: credential.id,
            rawId: credential.id,
            type: credential.type,
            response: {
                attestationObject: attestationObjectB64,
                clientDataJSON: clientDataJSONB64,
            },
        }),
        {
            friendlyName,
            sessionID,
        },
        {
            "X-Auth-Token": token,
        },
    );
    return await response.data;
};

/**
 * Return `true` if the given {@link redirectURL} (obtained from the redirect
 * query parameter passed around during the passkey verification flow) is one of
 * the whitelisted URLs that we allow redirecting to on success.
 */
export const isWhitelistedRedirect = (redirectURL: URL) =>
    (isDevBuild && redirectURL.hostname.endsWith("localhost")) ||
    redirectURL.host.endsWith(".ente.io") ||
    redirectURL.host.endsWith(".ente.sh") ||
    redirectURL.protocol == "ente:" ||
    redirectURL.protocol == "enteauth:";

export interface BeginPasskeyAuthenticationResponse {
    ceremonySessionID: string;
    options: Options;
}

interface Options {
    publicKey: PublicKeyCredentialRequestOptions;
}

export const beginPasskeyAuthentication = async (
    sessionId: string,
): Promise<BeginPasskeyAuthenticationResponse> => {
    try {
        const data = await HTTPService.post(
            `${ENDPOINT}/users/two-factor/passkeys/begin`,
            {
                sessionID: sessionId,
            },
        );

        return data.data;
    } catch (e) {
        log.error("begin passkey authentication failed", e);
        throw e;
    }
};

export const finishPasskeyAuthentication = async (
    credential: Credential,
    sessionId: string,
    ceremonySessionId: string,
) => {
    try {
        const data = await HTTPService.post(
            `${ENDPOINT}/users/two-factor/passkeys/finish`,
            {
                id: credential.id,
                rawId: credential.id,
                type: credential.type,
                response: {
                    authenticatorData: await toB64URLSafeNoPadding(
                        // eslint-disable-next-line @typescript-eslint/ban-ts-comment
                        // @ts-ignore
                        new Uint8Array(credential.response.authenticatorData),
                    ),
                    clientDataJSON: await toB64URLSafeNoPadding(
                        // eslint-disable-next-line @typescript-eslint/ban-ts-comment
                        // @ts-ignore
                        new Uint8Array(credential.response.clientDataJSON),
                    ),
                    signature: await toB64URLSafeNoPadding(
                        // eslint-disable-next-line @typescript-eslint/ban-ts-comment
                        // @ts-ignore
                        new Uint8Array(credential.response.signature),
                    ),
                    userHandle: await toB64URLSafeNoPadding(
                        // eslint-disable-next-line @typescript-eslint/ban-ts-comment
                        // @ts-ignore
                        new Uint8Array(credential.response.userHandle),
                    ),
                },
            },
            {
                sessionID: sessionId,
                ceremonySessionID: ceremonySessionId,
            },
        );

        return data.data;
    } catch (e) {
        log.error("finish passkey authentication failed", e);
        throw e;
    }
};

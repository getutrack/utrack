"use client";

import { Copy } from "lucide-react";
import { IApiToken } from "@utrack/types";
// ui
import { Button, Tooltip, TOAST_TYPE, setToast } from "@utrack/ui";
// helpers
import { renderFormattedDate } from "@/helpers/date-time.helper";
import { copyTextToClipboard } from "@/helpers/string.helper";
// types
import { usePlatformOS } from "@/hooks/use-platform-os";
// hooks

type Props = {
  handleClose: () => void;
  tokenDetails: IApiToken;
};

export const GeneratedTokenDetails: React.FC<Props> = (props) => {
  const { handleClose, tokenDetails } = props;
  const { isMobile } = usePlatformOS();
  const copyApiToken = (token: string) => {
    copyTextToClipboard(token).then(() =>
      setToast({
        type: TOAST_TYPE.SUCCESS,
        title: "Success!",
        message: "Token copied to clipboard.",
      })
    );
  };

  return (
    <div className="w-full p-5">
      <div className="w-full space-y-3 text-wrap">
        <h3 className="text-lg font-medium leading-6 text-custom-text-100">Key created</h3>
        <p className="text-sm text-custom-text-400">
          Copy and save this secret key in Utrack Pages. You can{"'"}t see this key after you hit Close. A CSV file
          containing the key has been downloaded.
        </p>
      </div>
      <button
        type="button"
        onClick={() => copyApiToken(tokenDetails.token ?? "")}
        className="mt-4 flex truncate w-full items-center justify-between rounded-md border-[0.5px] border-custom-border-200 px-3 py-2 text-sm font-medium outline-none"
      >
        <span className="truncate pr-2">{tokenDetails.token}</span>
        <Tooltip tooltipContent="Copy secret key" isMobile={isMobile}>
          <Copy className="h-4 w-4 text-custom-text-400 flex-shrink-0" />
        </Tooltip>
      </button>
      <div className="mt-6 flex items-center justify-between">
        <p className="text-xs text-custom-text-400">
          {tokenDetails.expired_at ? `Expires ${renderFormattedDate(tokenDetails.expired_at)}` : "Never expires"}
        </p>
        <Button variant="neutral-primary" size="sm" onClick={handleClose}>
          Close
        </Button>
      </div>
    </div>
  );
};

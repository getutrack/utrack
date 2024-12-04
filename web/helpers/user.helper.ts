import { EUserPermissions } from "@/youtrack-web/constants/user-permissions";

export const getUserRole = (role: EUserPermissions) => {
  switch (role) {
    case EUserPermissions.GUEST:
      return "GUEST";
    case EUserPermissions.MEMBER:
      return "MEMBER";
    case EUserPermissions.ADMIN:
      return "ADMIN";
  }
};

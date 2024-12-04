import { IState } from "@youtrack/types";
import { API_BASE_URL } from "@/helpers/common.helper";
import { APIService } from "./api.service";

export class StateService extends APIService {
  constructor() {
    super(API_BASE_URL);
  }

  async getStates(anchor: string): Promise<IState[]> {
    return this.get(`/api/public/anchor/${anchor}/states/`)
      .then((response) => response?.data)
      .catch((error) => {
        throw error?.response?.data;
      });
  }
}

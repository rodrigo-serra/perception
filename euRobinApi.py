#!/usr/bin/env python3

import requests, json

class EuRobinRESTAPI():
    def __init__(self):
        self.access_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJleHAiOjE2ODY5MzA1MzQuMTg1MjM3LCJwZXJtaXNzaW9ucyI6eyJldXJvYmluIjpbImdldF9kZW1vX3N0YXRlIiwic2V0X2RlbW9fc3RhdGVfaXN0IiwiZ2V0X3VzZCIsInNldF91c2QiXX19.Xgtl8QFZ_EJ9v4SpzY0JaXwm33F_b159yHKyDLIe2Kw"
        self.api_url = "https://eurobin.h2t.iar.kit.edu/api/v1/eurobin/demo-state/"

        self.phases = {
            1: {
                "name": "phase_01_fillParcel",
                "description": "Putting objects inside the parcel with human help",
                "responsible_partner": "sorbonne"
            },
            2: {
                "name": "phase_02_wrapParcel",
                "description": "Wrapping up the parcel",
                "responsible_partner": "sorbonne"
            },
            3: {
                "name": "phase_03_bringParcelToWP3",
                "description": "Picking and placing the parcel to be taken to the WP3 arena",
                "responsible_partner": "iit"
            },
            4: {
                "name": "phase_04_pickParcel",
                "description": "Picking the parcel",
                "responsible_partner": "use"
            },
            5: {
                "name": "phase_05_placeParcel",
                "description": "Placing the parcel in Swiss-Mile",
                "responsible_partner": "use"
            },
            6: {
                "name": "phase_06_takeParcelOutside",
                "description": "Placing the parcel in Swiss-Mile",
                "responsible_partner": "ethz"
            },
            7: {
                "name": "phase_07_bringParcelToWP2",
                "description": "Giving the parcel to be taken to the WP2 arena",
                "responsible_partner": "dlr"
            },
            8: {
                "name": "phase_08_takeParcelToDoor",
                "description": "Taking the parcel to the door of WP2 arena",
                "responsible_partner": "ist"
            },
            9: {
                "name": "phase_09_openDoor",
                "description": "Opening the door",
                "responsible_partner": "inria"
            },
            10: {
                "name": "phase_10_enterDoor",
                "description": "Entering the door",
                "responsible_partner": "ist"
            },
            11: {
                "name": "phase_11_takeParcelToTable",
                "description": "Taking the parcel to the table",
                "responsible_partner": "kit"
            },
            12: {
                "name": "phase_12_openParcel",
                "description": "Opening the parcel with a cutter",
                "responsible_partner": "kit"
            },
            13: {
                "name": "phase_13_emptyParcel",
                "description": "Taking objects from inside the parcel to the kitchen",
                "responsible_partner": ["kit", "dlr", "inria", "ist"]
            },
        }

    def _getRequest(self, api_url):
        headers =  {"access-token": self.access_token}
        response = requests.get(api_url, headers=headers)

        status_code = response.status_code
        
        if status_code == 200:
            return response.json()
        elif status_code == 401:
            print("PUT request: No token was supplied or the supplied token is invalid!")
            return False
        elif status_code == 403:
            print("PUT request: The supplied token lacks required permissions!")
            return False
        else:
            print("PUT request: Status code is unknow " + str(status_code))
            return False

    
    def _putRequest(self, api_url):
        headers =  {"access-token": self.access_token}
        response = requests.put(api_url, headers=headers, json={})
        
        status_code = response.status_code
        
        if status_code == 204:
            return True
        elif status_code == 400:
            print("PUT request: Invalid phase name or action!")
            return False
        elif status_code == 401:
            print("PUT request: No token was supplied or the supplied token is invalid!")
            return False
        elif status_code == 403:
            print("PUT request: The supplied token lacks required permissions!")
            return False
        else:
            print("PUT request: Status code is unknow " + str(status_code))
            return False
    

    def getDemoState(self):
        return self._getRequest(self.api_url)

    
    def getPartnerPhaseState(self, phase_code_num):
        status = self.getDemoState()
        if not status:
            return None

        phase_name = self.phases[phase_code_num]["name"]
        
        if not(phase_name in list(status.keys())):
            print("The phase name" + phase_name + " is not correct!")
            return None
        
        print(phase_name + " ready status: " + str(status[phase_name]["ready"]))
        return status[phase_name]["ready"]

    
    def startPhase(self, phase_code_num):
        phase_name = self.phases[phase_code_num]["name"]
        partner = self.phases[phase_code_num]["responsible_partner"]
        api_url = self.api_url + "partners/" + partner + "/phases/" + phase_name + "/started/"
        status = self._putRequest(api_url)
        if status:
            print("Started PUT request for " + phase_name + " was successfuly sent!")
        return status


    def finishPhase(self, phase_code_num):
        phase_name = self.phases[phase_code_num]["name"]
        partner = self.phases[phase_code_num]["responsible_partner"]
        api_url = self.api_url + "partners/" + partner + "/phases/" + phase_name + "/finished/"
        status = self._putRequest(api_url)
        if status:
            print("Finished PUT request for " + phase_name + " was successfuly sent!")
        return status


    def getIstTakeParcelToDoorState(self):
        return self.getPartnerPhaseState(8)

    def startIstTakeParcelToDoor(self):
        return self.startPhase(8)

    def finishIstTakeParcelToDoor(self):
        return self.finishPhase(8)

    def getIstEnterDoorState(self):
        return self.getPartnerPhaseState(10)

    def startIstEnterDoorPhase(self):
        return self.startPhase(10)

    def finishIstEnterDoorPhase(self):
        return self.finishPhase(10)

    def getKitOpenParcelState(self):
        return self.getPartnerPhaseState(12)
    


def main():
    api = EuRobinRESTAPI()
    print(api.getDemoState())
    # print(api.getIstTakeParcelToDoorState())
    # print(api.startIstTakeParcelToDoor())
    # print(api.finishIstTakeParcelToDoor())
    # print(api.getKitOpenParcelState())


if __name__ == '__main__':
    main()